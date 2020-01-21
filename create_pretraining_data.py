# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
# coding=utf-8
"""Create masked LM/next sentence masked_lm TF examples for ALBERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
import tokenization
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf
from itertools import combinations
import json
import statistics


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", True,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_bool(
    "favor_shorter_ngram", False,
    "Whether to set higher probabilities for sampling shorter ngrams.")

flags.DEFINE_bool(
    "random_next_sentence", False,
    "Whether to use the sentence that's right before the current sentence "
    "as the negative sample for next sentence prection, rather than using "
    "sentences from other random documents.")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("ngram", 3, "Maximum number of ngrams to mask.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 40,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

flags.DEFINE_bool(
    "do_hydro", True,
    "Whether or not to use local hydrophobicity predictions in training.")

flags.DEFINE_bool(
    "do_charge", True,
    "Whether or not to use local charge predictions in training.")

flags.DEFINE_bool(
    "do_pks", True,
    "Whether or not to use local predictions of pKa NH2, pKa COOH in training.")

flags.DEFINE_bool(
    "do_solubility", True,
    "Whether or not to use local predictions of solubility in training.")

flags.DEFINE_string(
  "aa_features", "./aa_features.json",
  "Location of AA features file to use in pretraining data creation.")

class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next, token_boundary):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.token_boundary = token_boundary
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels
    self.hydrophobicities = hydrophobicities
    self.charges = charges
    self.pks = pks
    self.solubilities = solubilities


  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "token_boundary: %s\n" % (" ".join(
        [str(x) for x in self.token_boundary]))
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "hydrophobicities: %s\n" % (" ".join(
        [str(x) for x in self.hydrophobicities]))
    s += "charges: %s\n" % (" ".join(
        [str(x) for x in self.charges]))
    s += "pks: %s\n" % (" ".join(
        [str(x) for x in self.pks]))
    s += "solubilities: %s\n" % (" ".join(
        [str(x) for x in self.solubilities]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    # token_boundary = list(instance.token_boundary)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      # token_boundary.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      hydrophobicities.append(0)
      solubilities.append(0)
      charges.append(0)
      pks.append(0)
      masked_lm_weights.append(0.0)
      hydrophobicity_weights.append(0.0)
      solubility_weights.append(0.0)
      charge_weights.append(0.0)
      pk_weights.append(0.0)


    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["token_boundary"] = create_int_feature(token_boundary)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["hydrophobicities"] = create_int_feature(hydrophobicities)
    features["solubilities"] = create_int_feature(solubilities)
    features["charges"] = create_int_feature(charges)
    features["pks"] = create_int_feature(pks)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["hydrophobicity_weights"] = create_float_feature(hydrophobicity_weights)
    features["solubility_weights"] = create_float_feature(solubility_weights)
    features["charge_weights"] = create_float_feature(charge_weights)
    features["pk_weights"] = create_float_feature(pk_weights)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)
    total_written += 1

    if inst_index < 2:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng, aa_features, 
                              do_hydro, do_charge, do_pks, do_solubility):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = reader.readline()
        if not FLAGS.spm_model_file:
          line = tokenization.convert_to_unicode(line)
        if not line:
          break
        if FLAGS.spm_model_file:
          line = tokenization.preprocess_text(line, lower=FLAGS.do_lower_case)
        else:
          line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng, aa_features, 
                              do_hydro, do_charge, do_pks, do_solubility))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, aa_features, 
                              do_hydro, do_charge, do_pks, do_solubility):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 2
  target_seq_length = max_num_tokens

  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  instances = []
  i = 0
  while i < len(document):
    if len(document[i]) == 0:
      print('> Doc[i] was empty, i = ', i)
      continue

    lost = len(document[i]) - target_seq_length
    tokens_a = document[i][:target_seq_length]

    if (len(tokens_a) == 0):
      print('index', i)
      print(document[i])
      i += 1
      continue

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    (tokens, masked_lm_positions, masked_lm_labels, token_boundary,
      hydrophobicities, charges, pks, solubilities) = create_local_predictions(
          tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, 
          aa_features, do_hydro, do_charge, do_pks, do_solubility)

    instance = TrainingInstance(
        tokens=tokens,
        segment_ids=segment_ids,
        token_boundary=token_boundary,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    if lost > 20:
      document[i] = document[i][target_seq_length:]
      continue

    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label", "hydrophobicity", "charge", "pks", "solubility"])


def _is_start_piece_sp(piece):
  """Check if the current word piece is the starting piece (sentence piece)."""
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  special_pieces.add(u"€".encode("utf-8"))
  special_pieces.add(u"£".encode("utf-8"))
  # Note(mingdachen):
  # For foreign characters, we always treat them as a whole piece.
  english_chars = set(list("abcdefghijklmnopqrstuvwxyz"))
  if (six.ensure_str(piece).startswith("▁") or
      six.ensure_str(piece).startswith("<") or piece in special_pieces or
      not all([i.lower() in english_chars.union(special_pieces)
               for i in piece])):
    return True
  else:
    return False


def _is_start_piece_bert(piece):
  """Check if the current word piece is the starting piece (BERT)."""
  # When a word has been split into
  # WordPieces, the first token does not have any marker and any subsequence
  # tokens are prefixed with ##. So whenever we see the ## token, we
  # append it to the previous set of word indexes.
  return not six.ensure_str(piece).startswith("##")


def is_start_piece(piece):
  if FLAGS.spm_model_file:
    return _is_start_piece_sp(piece)
  else:
    return _is_start_piece_bert(piece)


def create_local_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 aa_features, do_hydro, do_charge, do_pks, do_solubility):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  # Note(mingdachen): We create a list for recording if the piece is
  # the starting piece of current token, where 1 means true, so that
  # on-the-fly whole word masking is possible.
  token_boundary = [0] * len(tokens)

  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      token_boundary[i] = 1
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        not is_start_piece(token)):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])
      if is_start_piece(token):
        token_boundary[i] = 1

  output_tokens = list(tokens)

  masked_lm_positions = []
  masked_lm_labels = []

  if masked_lm_prob == 0:
    return (output_tokens, masked_lm_positions,
            masked_lm_labels, token_boundary)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  # Note(mingdachen):
  # By default, we set the probilities to favor longer ngram sequences.
  ngrams = np.arange(1, FLAGS.ngram + 1, dtype=np.int64)
  pvals = 1. / np.arange(1, FLAGS.ngram + 1)
  pvals /= pvals.sum(keepdims=True)

  if FLAGS.favor_shorter_ngram:
    pvals = pvals[::-1]

  ngram_indexes = []
  for idx in range(len(cand_indexes)):
    ngram_index = []
    for n in ngrams:
      ngram_index.append(cand_indexes[idx:idx+n])
    ngram_indexes.append(ngram_index)

  rng.shuffle(ngram_indexes)

  masked_lms = []
  covered_indexes = set()
  for cand_index_set in ngram_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if not cand_index_set:
      continue
    # Note(mingdachen):
    # Skip current piece if they are covered in lm masking or previous ngrams.
    for index_set in cand_index_set[0]:
      for index in index_set:
        if index in covered_indexes:
          continue

    n = np.random.choice(ngrams[:len(cand_index_set)],
                         p=pvals[:len(cand_index_set)] /
                         pvals[:len(cand_index_set)].sum(keepdims=True))
    index_set = sum(cand_index_set[n - 1], [])
    n -= 1
    # Note(mingdachen):
    # Repeatedly looking for a candidate that does not exceed the
    # maximum number of predictions by trying shorter ngrams.
    while len(masked_lms) + len(index_set) > num_to_predict:
      if n == 0:
        break
      index_set = sum(cand_index_set[n - 1], [])
      n -= 1
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      original_token = tokens[index]
      
      hydrophobicity = get_hydrophobicity(original_token, aa_features) if do_hydro else 0
      charge = get_charge(original_token, aa_features) if do_charge else 0
      pks = get_pks(original_token, aa_features) if do_pks else 0
      solubility = get_solubility(original_token, aa_features) if do_solubility else 0

      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index], hydrophobicity=hydrophobicity, charge=charge, pks=pks, solubility=solubility))
  assert len(masked_lms) <= num_to_predict

  rng.shuffle(ngram_indexes)

  select_indexes = set()
  masked_lms = sorted(masked_lms, key=lambda x: x.index)


  masked_lm_positions = [p.index for p in masked_lms]
  masked_lm_labels = [p.label for p in masked_lms]
  hydrophobicities = [p.hydrophobicity for p in masked_lms] if do_hydro else None
  charges = [p.charge for p in masked_lms] if do_charge else None
  pks = [p.pks for p in masked_lms] if do_pks else None
  solubilities = [p.solubility for p in masked_lms] if do_solubility else None

  return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, ydrophobicities, charges, pks, solubilities)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def get_hydrophobicity(peptide, aa_features):
    k = len(peptide)
    values = [feats["hydrophobicity"] for feats in aa_features.values()]
    kmer_values = [sum(k_values) for k_values in combinations(values, k)]
    lower_bound = np.percentile(kmer_values, 33.33)
    upper_bound = np.percentile(kmer_values, 66.67)  
    print(lower_bound, upper_bound)
    DEFAULT_GUESS = statistics.median(kmer_values)

    res = 0
    for amino_acid in peptide:
        if amino_acid in aa_features:
            res += aa_features[amino_acid]["hydrophobicity"]
        else:
            res += DEFAULT_GUESS
    if res < lower_bound:
        return 0
    elif res < upper_bound:
        return 1
    else:
        return 2

def get_charge(peptide, aa_features):
    k = len(peptide)
    values = [feats["charge"] for feats in aa_features.values()]
    kmer_values = [sum(k_values) for k_values in combinations(values, k)]
    lower_bound = np.percentile(kmer_values, 33.33)
    upper_bound = np.percentile(kmer_values, 66.67)  
    print(lower_bound, upper_bound)
    DEFAULT_GUESS = statistics.median(kmer_values)

    res = 0
    for amino_acid in peptide:
        if amino_acid in aa_features:
            res += aa_features[amino_acid]["charge"]
        else:
            res += DEFAULT_GUESS
    if res < lower_bound:
        return 0
    elif res < upper_bound:
        return 1
    else:
        return 2

def get_pks(peptide, aa_features):
    k = len(peptide)
    values = [feats["pks"] for feats in aa_features.values()]
    kmer_values = [sum(k_values) for k_values in combinations(values, k)]
    lower_bound = np.percentile(kmer_values, 33.33)
    upper_bound = np.percentile(kmer_values, 66.67)  
    print(lower_bound, upper_bound)
    DEFAULT_GUESS = statistics.median(kmer_values)

    res = 0
    for amino_acid in peptide:
        if amino_acid in aa_features:
            res += aa_features[amino_acid]["pks"]
        else:
            res += DEFAULT_GUESS
    if res < lower_bound:
        return 0
    elif res < upper_bound:
        return 1
    else:
        return 2

def get_solubility(peptide, aa_features):
    k = len(peptide)
    values = [feats["solubility"] for feats in aa_features.values()]
    kmer_values = [sum(k_values) for k_values in combinations(values, k)]
    lower_bound = np.percentile(kmer_values, 33.33)
    upper_bound = np.percentile(kmer_values, 66.67)  
    print(lower_bound, upper_bound)
    DEFAULT_GUESS = statistics.median(kmer_values)

    res = 0
    for amino_acid in peptide:
        if amino_acid in aa_features:
            res += aa_features[amino_acid]["solubility"]
        else:
            res += DEFAULT_GUESS
    if res < lower_bound:
        return 0
    elif res < upper_bound:
        return 1
    else:
        return 2




def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  with open(FLAGS.aa_features, "r") as aa_feature_file:
    aa_feature_text = aa_feature_file.read()
  aa_features = json.loads(aa_feature_text)


  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng, aa_features, 
      FLAGS.do_hydro, FLAGS.do_charge, FLAGS.do_pks, FLAGS.do_solubility)

  tf.logging.info("number of instances: %i", len(instances))

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
