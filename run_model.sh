python3 models.py \
/data/semeval/training/articles-training-20180831.spacy_links.xml \
/data/semeval/training/ground-truth-training-20180831.xml \
--train_size 50 \
--article_length 100 \
--model 0 \
--epoch 60 \
--batch 20 \
--test_data /data/semeval/validation/articles-validation-20180831.spacy_links.xml \
--test_labels /data/semeval/validation/ground-truth-validation-20180831.xml \
--test_size 5
