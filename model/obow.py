import copy
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import obow.utils as utils
import obow.solver as solver
import obow.fewshot as fewshot
from obow.classification import PredictionHead




class OBoW(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_channels,
        bow_levels,
        bow_extractor_opts_list,
        bow_predictor_opts,
        alpha=0.99,
        num_classes=None,
    ):
        """Builds an OBoW model.
        Args:
        feature_extractor: essentially the convnet model that is going to be
            trained in order to learn image representations.
        num_channels: number of channels of the output global feature vector of
            the feature_extractor.
        bow_levels: a list with the names (strings) of the feature levels from
            which the teacher network in OBoW will create BoW targets.
        bow_extractor_opts_list: a list of dictionaries with the configuration
            options for the BoW extraction (at teacher side) for each BoW level.
            Each dictionary should define the following keys (1) "num_words"
            with the vocabulary size of this level, (2) "num_channels",
            optionally (3) "update_type" (default: "local_averaging"),
            optionally (4) "inv_delta" (default: 15), which is the inverse
            temperature that is used for computing the soft assignment codes,
            and optionally (5) "bow_pool" (default: "max"). For more details
            see the documentation of the BoWExtractor class.
        bow_predictor_opts: a dictionary with configuration options for the
            BoW prediction head of the student. The dictionary must define
            the following keys (1) "kappa", a coefficent for scaling the
            magnitude of the predicted weights, and optionally (2) "learn_kappa"
            (default: False),  a boolean value that if true kappa becomes a
            learnable parameter. For all the OBoW experiments "learn_kappa" is
            set to False. For more details see the documentation of the
            BoWPredictor class.
        alpha: the momentum coefficient between 0.0 and 1.0 for the teacher
            network updates. If alpha is a scalar (e.g., 0.99) then a static
            momentum coefficient is used during training. If alpha is tuple of
            two values, e.g., alpha=(alpha_base, num_iterations), then OBoW
            uses a cosine schedule that starts from alpha_base and it increases
            it to 1.0 over num_iterations.
        num_classes: (optional) if not None, then it creates a
            linear classification head with num_classes outputs that would be
            on top of the teacher features for on-line monitoring the quality
            of the learned features. No gradients would back-propagated from
            this head to the feature extractor trunks. So, it does not
            influence the learning of the feature extractor. Note, at the end
            the features that are used are those of the student network, not
            of the teacher.
        """
        super(OBoW, self).__init__()
        assert isinstance(bow_levels, (list, tuple))
        assert isinstance(bow_extractor_opts_list, (list, tuple))
        assert len(bow_extractor_opts_list) == len(bow_levels)

        self._bow_levels = bow_levels
        self._num_bow_levels = len(bow_levels)
        if isinstance(alpha, (tuple, list)):
            # Use cosine schedule in order to increase the alpha from
            # alpha_base (e.g., 0.99) to 1.0.
            alpha_base, num_iterations = alpha
            self._alpha_base = alpha_base
            self._num_iterations = num_iterations
            self.register_buffer("_alpha", torch.FloatTensor(1).fill_(alpha_base))
            self.register_buffer("_iteration", torch.zeros(1))
            self._alpha_cosine_schedule = True
        else:
            self._alpha = alpha
            self._alpha_cosine_schedule = False

        # Build the student network components.
        self.feature_extractor = feature_extractor
        assert "kappa" in bow_predictor_opts
        bow_predictor_opts["num_channels_out"] = num_channels
        bow_predictor_opts["num_channels_hidden"] = num_channels * 2
        bow_predictor_opts["num_channels_in"] = [
            d["num_channels"] for d in bow_extractor_opts_list]
        self.bow_predictor = BoWPredictor(**bow_predictor_opts)

        # Build the teacher network components.
        self.feature_extractor_teacher = copy.deepcopy(self.feature_extractor)
        self.bow_extractor = BoWExtractorMultipleLevels(bow_extractor_opts_list)

        if (num_classes is not None):
            self.linear_classifier = PredictionHead(
                num_channels=num_channels, num_classes=num_classes,
                batch_norm=True, pool_type="global_avg")
        else:
            self.linear_classifier = None

        for param, param_teacher in zip(
            self.feature_extractor.parameters(),
            self.feature_extractor_teacher.parameters()):
            param_teacher.data.copy_(param.data)  # initialize
            param_teacher.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _get_momentum_alpha(self):
        if self._alpha_cosine_schedule:
            scale = 0.5 * (1. + math.cos((math.pi * self._iteration.item()) / self._num_iterations))
            self._alpha.fill_(1.0 - (1.0 - self._alpha_base) * scale)
            self._iteration += 1
            return self._alpha.item()
        else:
            return self._alpha

    @torch.no_grad()
    def _update_teacher(self):
        """ Exponetial moving average for the feature_extractor_teacher params:
            param_teacher = param_teacher * alpha + param * (1-alpha)
        """
        if not self.training:
            return
        alpha = self._get_momentum_alpha()
        if alpha >= 1.0:
            return
        for param, param_teacher in zip(
            self.feature_extractor.parameters(),
            self.feature_extractor_teacher.parameters()):
            param_teacher.data.mul_(alpha).add_(
                param.detach().data, alpha=(1. - alpha))

    def _bow_loss(self, bow_prediction, bow_target):
        assert isinstance(bow_prediction, (list, tuple))
        assert isinstance(bow_target, (list, tuple))
        assert len(bow_prediction) == self._num_bow_levels
        assert len(bow_target) == self._num_bow_levels

        # Instead of using a custom made cross-entropy loss for soft targets,
        # we use the pytorch kl-divergence loss that is defined as the
        # cross-entropy plus the entropy of targets. Since there is no gradient
        # back-propagation from the targets, it is equivalent to cross entropy.
        loss = [
            F.kl_div(F.log_softmax(p, dim=1), expand_target(t, p), reduction="batchmean")
            for (p, t) in zip(bow_prediction, bow_target)]
        return torch.stack(loss).mean()

    def _linear_classification(self, features, labels):
        # With .detach() no gradients of the classification loss are
        # back-propagated to the feature extractor.
        # The reason for training such a linear classifier is in order to be
        # able to monitor while training the quality of the learned features.
        features = features.detach()
        if (labels is None) or (self.linear_classifier is None):
            return (features.new_full((1,), 0.0).squeeze(),
                    features.new_full((1,), 0.0).squeeze())

        scores = self.linear_classifier(features)
        loss = F.cross_entropy(scores, labels)
        with torch.no_grad():
            accuracy = utils.top1accuracy(scores, labels).item()

        return loss, accuracy

    def generate_bow_targets(self, image):
        features = self.feature_extractor_teacher(image, self._bow_levels)
        if isinstance(features, torch.Tensor):
            features = [features,]
        bow_target, _ = self.bow_extractor(features)
        return bow_target, features

    def forward_test(self, img_orig, labels):
        with torch.no_grad():
            features = self.feature_extractor_teacher(img_orig, self._bow_levels)
            features = features if isinstance(features, torch.Tensor) else features[-1]
            features = features.detach()
            loss_cls, accuracy = self._linear_classification(features, labels)

        return loss_cls, accuracy

    def forward(self, img_orig, img_crops, labels=None):
        """ Applies the OBoW self-supervised task to a mini-batch of images.
        Args:
        img_orig: 4D tensor with shape [batch_size x 3 x img_height x img_width]
            with the mini-batch of images from which the teacher network
            generates the BoW targets.
        img_crops: list of 4D tensors where each of them is a mini-batch of
            image crops with shape [(batch_size * num_crops) x 3 x crop_height x crop_width]
            from which the student network predicts the BoW targets. For
            example, in the full version of OBoW this list will iclude a
            [(batch_size * 2) x 3 x 160 x 160]-shaped tensor with two image crops
            of size [160 x 160] pixels and a [(batch_size * 5) x 3 x 96 x 96]-
            shaped tensor with five image patches of size [96 x 96] pixels.
        labels: (optional) 1D tensor with shape [batch_size] with the class
            labels of the img_orig images. If available, it would be used for
            on-line monitoring the performance of the linear classifier.
        Returns:
        losses: a tensor with the losses for each type of image crop and
            (optionally) the loss of the linear classifier.
        logs: a list of metrics for monitoring the training progress. It
            includes the perplexity of the bow targets in a mini-batch
            (perp_b), the perplexity of the bow targets in an image (perp_i),
            and (optionally) the accuracy of a linear classifier on-line
            trained on the teacher features (this is a proxy for monitoring
            during training the quality of the learned features; Note, at the
            end the features that are used are those of the student).
        """
        if self.training is False:
            # For testing, it only computes the linear classification accuracy.
            return self.forward_test(img_orig, labels)

        #*********************** MAKE BOW PREDICTIONS **************************
        dictionary = self.bow_extractor.get_dictionary()
        features = [self.feature_extractor(x) for x in img_crops]
        bow_predictions = self.bow_predictor(features, dictionary)
        #***********************************************************************
        #******************** COMPUTE THE BOW TARGETS **************************
        with torch.no_grad():
            self._update_teacher()
            bow_target, features_t = self.generate_bow_targets(img_orig)
            perp_b, perp_i = compute_bow_perplexity(bow_target)
        #***********************************************************************
        #***************** COMPUTE THE BOW PREDICTION LOSSES *******************
        losses = [self._bow_loss(pred, bow_target) for pred in bow_predictions]
        #***********************************************************************
        #****** MONITORING: APPLY LINEAR CLASSIFIER ON TEACHER FEATURES ********
        loss_cls, accuracy = self._linear_classification(features_t[-1], labels)
        #***********************************************************************

        losses = torch.stack(losses + [loss_cls,], dim=0).view(-1)
        logs = list(perp_b + perp_i) + [accuracy,]

        return losses, logs