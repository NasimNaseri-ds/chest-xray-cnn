import pandas as pd
import numpy as np
import os
import tensorflow as tf


# load csv
df_path = '/data/Data_Entry_2017_v2020.csv'
df = pd.read_csv(df_path)
df.head()


# In[6]:deleting extra rows from csv
images_path = '/data/pictures'
available_images = set(os.listdir(images_path))
df_filtered = df[df['Image Index'].isin(available_images)].reset_index(drop=True)
print("Original CSV rows:", len(df))
print("Rows after filtering:", len(df_filtered))


# In[10]:cheking for duplicates or NaN
duplicates = df_filtered[df_filtered.duplicated(subset=['Image Index'], keep=False)]
print("Number of duplicate rows:", len(duplicates))
print(duplicates.head())
missing = df_filtered[df_filtered['Finding Labels'].isna() | (df_filtered['Finding Labels'].str.strip() == '')]
print("Number of rows with empty/missing labels:", len(missing))


#deleting extra columns
df_filtered.head()
df_filtered = df_filtered[["Image Index", "Finding Labels"]]  
df_filtered.head()


# In[17]:chek gcategories


all_labels = df_filtered["Finding Labels"].str.split("|").explode()
label_counts = all_labels.value_counts()
print("Number of unique diseases:", label_counts.shape[0])
print("\nCounts per disease:\n", label_counts)
df_expanded = df_filtered.assign(Finding_Labels=df["Finding Labels"].str.split("|")).explode("Finding_Labels")

print(df_expanded.shape)
label_counts = df_expanded["Finding_Labels"].value_counts()
print("Number of unique diseases:", label_counts.shape[0])
print("\nCounts per disease:\n", label_counts)


# In[24]:reduce categoreis
threshold = 500
rare_diseases = label_counts[label_counts < threshold].index.tolist()
print("Rare diseases:", rare_diseases)
def categorize(label):
    if label == "No Finding":
        return "Healthy"
    elif label in rare_diseases:
        return "Rare disease"
    else:
        return "Normal disease"
df_expanded["Category"] = df_expanded["Finding_Labels"].apply(categorize)
df_final = df_expanded.groupby("Image Index")["Category"].apply(
    lambda x: "Rare disease" if "Rare disease" in x.values
              else ("Normal disease" if "Normal disease" in x.values
                    else "Healthy")
).reset_index()
df_final.to_csv("dffiltered_3categories.csv", index=False)
df_final.head()


# In[31]:check balanc
category_counts = df_final["Category"].value_counts()
print(category_counts)
df_final["filepath"] = df_final["Image Index"].apply(lambda x: os.path.join(images_path, x))


# In[34]:name of columns
df_final = df_final[["filepath", "Category"]]
print(df_final.head())


# In[36]:split data
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(
    df_final,
    test_size=0.3,
    stratify=df_final["Category"],
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["Category"],
    random_state=42
)
print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))
print("Train counts:\n", train_df["Category"].value_counts())
print("Validation counts:\n", val_df["Category"].value_counts())
print("Test counts:\n", test_df["Category"].value_counts())


# In[41]:rescale with ImageGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_generator=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.3,
)
image_generator_test=ImageDataGenerator(
    rescale=1./255
)
df_final["ext"] = df_final["filepath"].str.lower().str.split(".").str[-1]
print(df_final["ext"].value_counts())


# In[45]:make image gen
from PIL import Image
train_gen = image_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="Category",
    target_size=(224,224),
    directory=None,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=50,
    shuffle=True
)
val_gen = image_generator_test.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepath",
    y_col="Category",
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32
)


# In[48]:callbacks
from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss',mode='min',patience=5)
from tensorflow.keras.callbacks import ReduceLROnPlateau
rlop=ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.2, min_lr=1e-6)
from tensorflow.keras.callbacks import ModelCheckpoint
mch=ModelCheckpoint('chestxray.keras',monitor='val_loss',mode='min',save_best_only=True)


# In[54]:Makin a cnn model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
base_model = DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)  # requires 3 channels
)
base_model.trainable = False
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 categories: normal, disease, rare
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


# In[60]:balance the data insid the model
from sklearn.utils.class_weight import compute_class_weight
y_train = train_gen.classes
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(zip(np.unique(y_train), weights))
print(class_weights)


# In[ ]:fit the model
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=50,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    callbacks=[es, rlop, mch],
    class_weight=class_weights
)
from tensorflow.keras.models import load_model
model = load_model("chestxray.keras")
test_gen = image_generator_test.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="Category",
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32
)
test_loss, test_acc = model.evaluate(test_gen, steps=len(test_gen))
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)
#end