"""
Partition class (holds feature information, feature values, and labels for a
dataset). Includes helper class Example.
Author: Sara Mathieson + Russell Gerhard
Date: 9/16/19
"""

import numpy as np

class Example:

    def __init__(self, features, label):
        """Helper class (like a struct) that stores info about each example."""
        # Dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {-1, 1}

class Partition:

    def __init__(self, data, F):
        """Store information about a dataset"""
        self.data = data # list of examples
        # Dictionary. key=feature name: value=set of possible values
        self.F = F
        self.n = len(self.data)

    def labProbability(self, lab_val):
        """Compute probability that label takes lab_val"""
        count = 0
        # Iterate over examples, counting when label takes lab_val
        for example in self.data:
            if (example.label == lab_val):
                count+=1
        return(count/self.n)

    def featProbability(self, feature, feat_val):
        """Compute probability that feature takes feat_val"""
        count = 0
        # Iterate over examples, counting when feature takes feat_val
        for example in self.data:
            if (example.features[feature] == feat_val):
                count+=1
        return(count/self.n)

    def condProbability(self, feature, feat_val, lab_val):
        """Compute conditional probability that label takes lab_val
           AND feature takes feat_val"""
        count = 0
        # Iterate over examples, counting when BOTH
        # feature takes feat_val AND
        # label takes lab_val
        for example in self.data:
            if (example.features[feature] == feat_val) and \
               (example.label == lab_val):
                count+=1

        # Use definition of cond. prob.
        numerator = count/self.n
        denominator = self.featProbability(feature, feat_val)
        if denominator==0:
            return 0
        else:
            return (numerator/denominator)

    def individualEntropy(self):
        """Compute entropy of the label"""
        entropy = 0

        # Sum over all label values using definition of entropy
        for lab_val in (-1,1):
            prob = self.labProbability(lab_val)
            log_prob = -np.log2(prob)
            entropy += prob*log_prob
        return(entropy)

    def condEntropy(self, feature):
        """Compute entropy of the label given a feature"""
        all_cond_entropy = 0 #total entropy given a feature

        # Iterate over all possible values of our feature
        # To sum up P(X=v)*H(Y|X=v) for all values of our feature
        for feat_val in self.F[feature]:
            feat_prob = self.featProbability(feature, feat_val)
            feat_cond_entropy = 0 # stores H(Y|X=v)

            # Iterate over LABEL values within each iteration of FEATURE values
            # In order to compute H(Y|X=v) for each value of our feature
            for lab_val in (-1,1):
                cond_prob = self.condProbability(feature, feat_val, lab_val)
                if cond_prob!=0:
                    log_cond_prob = -np.log2(cond_prob)
                else:
                    log_cond_prob = 0
                feat_cond_entropy += cond_prob*log_cond_prob

            # Add P(X=v)*H(Y|X=v) to total entropy for each v
            all_cond_entropy += feat_prob*feat_cond_entropy

        return(all_cond_entropy)

    def infoGain(self, feature):
        """Compute information gain of a feature"""
        return(self.individualEntropy() - self.condEntropy(feature))

    def getMaxGainFeat(self):
        """Return the feature that maximizes information gain"""
        maxGain = 0
        # Iterate over features, compute infoGain and return feature w/ max gain
        for feature in self.F.keys():
            if(self.infoGain(feature) > maxGain):
                maxGain = self.infoGain(feature)
                out_feature = feature
        return out_feature
