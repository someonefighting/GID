#!/bin/bash

. ./cmd.sh
. ./path.sh

set -e
# testset should be prepard in data/testset with valid utt2spk, spk2utt, utt2gender, wav.scp
testset=dongchedi

# default model
num_components=128 # Larger than this doesn't make much of a difference.
ivector_dim=256

plpdir=`pwd`/plp_pitch
vaddir=`pwd`/plp_pitch
data=data/${testset}
nj=24

## feature extraction
#steps/make_plp_pitch.sh --nj $nj --cmd "$train_cmd" \
#   $data exp/make_plp_pitch $plpdir
#
# utils/fix_data_dir.sh $data
#
# sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
#   $data exp/make_vad_pitch $vaddir

## ivector extraction and logistic regression
sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
  exp/extractor_pitch $data \
  exp/ivectors_${testset}_pitch

lid/run_logistic_regression.sh --prior-scale 0.0 \
  --conf conf/logistic-regression.conf --train_dir exp/ivectors_train_pitch \
  --test_dir exp/ivectors_${testset}_pitch --model_dir exp/ivectors_train_pitch --test_utt2lang $data/utt2gender

