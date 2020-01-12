#coding=utf-8
from common_lib import *
num_filters = 32


path = '../data/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE = '../data/glove.6B.50d.txt'; embed_size = 50  # how big is each word vector
# EMBEDDING_FILE = '../data/glove.840B.300d.txt';embed_size = 300  # how big is each word vector
# EMBEDDING_FILE = '../data/fasttext.300d.txt';embed_size = 300  # how big is each word vector
TRAIN_DATA_FILE = '../data/train_raw.csv'
TEST_DATA_FILE = '../data/test_raw.csv'

max_features = 9000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 200  # max number of words in a comment to use
number_filters = 32  # the number of CNN filters
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
print 'train data', train.shape
print 'test data', test.shape

list_sentences_train = train["comment_text"].fillna("_na_").values
list_sentences_test = test["comment_text"].fillna("_na_").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
y_te = test[list_classes].values

special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
replace_numbers = re.compile(r'\d+', re.IGNORECASE)


def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    # Remove Special Characters
    text = special_character_removal.sub('', text)

    # Replace Numbers
    text = replace_numbers.sub('n', text)
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


print ('step 1: preprogress')
comments = []
for text in list_sentences_train:
    comments.append(text_to_wordlist(text))

test_comments = []
for text in list_sentences_test:
    test_comments.append(text_to_wordlist(text))

tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
tokenizer.fit_on_texts(list(comments + test_comments))

comments_sequence = tokenizer.texts_to_sequences(comments)
test_comments_sequence = tokenizer.texts_to_sequences(test_comments)

X_t = pad_sequences(comments_sequence, maxlen=maxlen)
X_te = pad_sequences(test_comments_sequence, maxlen=maxlen)

# X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
# X_te = X_te.reshape((X_te.shape[0], 1, X_te.shape[1]))

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

#词嵌入
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print ('step 2: net')
def get_model():
    filter_sizes = [1, 2, 3, 5]
    num_filters = 32
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, embed_size, 1))(x)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                    activation='elu')(x)

    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(len(list_classes), activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model()

print ('step 3: fit')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X_t, y, batch_size=64, epochs=100, callbacks=[early_stopping], validation_data=(X_te, y_te))


print ('save model')
model.save('../data/model/textcnn_model_cnv.best')
model = keras.models.load_model('../data/model/textcnn_model_cnv.best')
from keras.utils import plot_model
plot_model(model, to_file='../data/network/textcnn.png', show_shapes=True)

print ('step 4: predict')
y_test = model.predict([X_te], batch_size=128, verbose=1)

sample_submission = test
sample_submission[list_classes] = y_test
sample_submission.to_csv('../data/submission_textcnn_cnv.csv', index=False)
