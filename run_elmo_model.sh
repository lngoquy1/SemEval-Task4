python3 elmo_models.py \
/data/semeval/training/articles-training-20180831.spacy_links.xml \
/data/semeval/training/ground-truth-training-20180831.xml \
--train_size 50 \
--testing /data/semeval/validation/articles-validation-20180831.spacy_links.xml \
--test_labels /data/semeval/validation/ground-truth-validation-20180831.xml \
--test_size 5
