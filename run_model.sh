python3 models.py \
/data/semeval/training/articles-training-20180831.spacy_links.xml \
/data/semeval/training/ground-truth-training-20180831.xml \
--train_size 50000 \
--article_length 550 \
--model 0 \
--epoch 60 \
--batch 20 \
--dropout_attention 0.0 \
--test_data /data/semeval/validation/articles-validation-20180831.spacy_links.xml \
--test_labels /data/semeval/validation/ground-truth-validation-20180831.xml \
--test_size 5000 > "output_lstm_glove.txt"
