#!/bin/bash

. ./cmd.sh
. ./path.sh

set -e
plpdir=`pwd`/plp
vaddir=`pwd`/plp
trials=data/test/trials
num_components=64 # Larger than this doesn't make much of a difference.
nj=24

 steps/make_plp.sh --nj $nj --cmd "$train_cmd" \
   data/dongchedi exp/make_plp $plpdir
 sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
   data/dongchedi exp/make_vad $vaddir
 sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
   exp/extractor data/dongchedi \
   exp/ivectors_dongchedi
echo "finish"
exit
# steps/make_plp.sh --nj $nj --cmd "$train_cmd" \
#   data/train exp/make_plp $plpdir
# steps/make_plp.sh --nj $nj --cmd "$train_cmd" \
#   data/test exp/make_plp $plpdir
#
# for name in train test; do
#   utils/fix_data_dir.sh data/${name}
# done
#
# sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
#   data/train exp/make_vad $vaddir
# sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
#   data/test exp/make_vad $vaddir
#
# for name in train test; do
#   utils/fix_data_dir.sh data/${name}
# done

# Train UBM and i-vector extractor.
sid/train_diag_ubm.sh --cmd "$train_cmd" \
  --nj $nj --num-threads $nj \
  data/train $num_components \
  exp/diag_ubm_$num_components

sid/train_full_ubm.sh --nj $nj --remove-low-count-gaussians false \
  --cmd "$train_cmd" data/train \
  exp/diag_ubm_$num_components exp/full_ubm_$num_components

sid/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj --num-threads 1 --num-processes 1\
  --ivector-dim 128 \
  --num-iters 5 exp/full_ubm_$num_components/final.ubm data/train \
  exp/extractor

# Extract i-vectors.
sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
  exp/extractor data/train \
  exp/ivectors_train

sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
  exp/extractor data/test \
  exp/ivectors_test

## Separate the i-vectors into male and female partitions and calculate
## i-vector means used by the scoring scripts.
#local/scoring_common.sh data/sre data/sre10_train data/sre10_test \
#  exp/ivectors_sre exp/ivectors_sre10_train exp/ivectors_sre10_test

# The commented out scripts show how to do cosine scoring with and without
# first reducing the i-vector dimensionality with LDA. PLDA tends to work
# best, so we don't focus on the scores obtained here.
#
local/cosine_scoring.sh data/train data/test \
  exp/ivectors_train exp/ivectors_test $trials exp/scores

## Create a gender independent PLDA model and do scoring.
#local/plda_scoring.sh data/sre data/sre10_train data/sre10_test \
#  exp/ivectors_sre exp/ivectors_sre10_train exp/ivectors_sre10_test $trials exp/scores_gmm_2048_ind_pooled
#local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_female data/sre10_test_female \
#  exp/ivectors_sre exp/ivectors_sre10_train_female exp/ivectors_sre10_test_female $trials_female exp/scores_gmm_2048_ind_female
#local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_male data/sre10_test_male \
#  exp/ivectors_sre exp/ivectors_sre10_train_male exp/ivectors_sre10_test_male $trials_male exp/scores_gmm_2048_ind_male
#
## Create gender dependent PLDA models and do scoring.
#local/plda_scoring.sh data/sre_female data/sre10_train_female data/sre10_test_female \
#  exp/ivectors_sre exp/ivectors_sre10_train_female exp/ivectors_sre10_test_female $trials_female exp/scores_gmm_2048_dep_female
#local/plda_scoring.sh data/sre_male data/sre10_train_male data/sre10_test_male \
#  exp/ivectors_sre exp/ivectors_sre10_train_male exp/ivectors_sre10_test_male $trials_male exp/scores_gmm_2048_dep_male
#
## Pool the gender dependent results.
#mkdir -p exp/scores_gmm_2048_dep_pooled
#cat exp/scores_gmm_2048_dep_male/plda_scores exp/scores_gmm_2048_dep_female/plda_scores \
#  > exp/scores_gmm_2048_dep_pooled/plda_scores

# GMM-2048 PLDA EER
# ind pooled: 2.26
# ind female: 2.33
# ind male:   2.05
# dep female: 2.30
# dep male:   1.59
# dep pooled: 2.00
echo "GMM-$num_components EER"
eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores/cosine_scores) 2> /dev/null`
echo $eer

