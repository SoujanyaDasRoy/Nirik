import os, glob, hashlib, random
import numpy as np
import pandas as pd
import tensorflow as tf
import pydicom
import cv2
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import layers, models, optimizers, losses, metrics

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

BASE = "/kaggle/input/datasets"
PATHS = {
    "montgomery":  f"{BASE}/raddar/tuberculosis-chest-xrays-montgomery/images",
    "shenzhen":    f"{BASE}/raddar/tuberculosis-chest-xrays-shenzhen/images",
    "tb_database": f"{BASE}/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database",
    "nirt_dicom":  f"{BASE}/projectmantra/nirt-india-chest-x-ray-dicom-dataset",
}
for k, v in PATHS.items():
    print(k, "->", "OK" if os.path.isdir(v) else "MISSING", v)
print("GPU:", tf.config.list_physical_devices('GPU'))

records = []

def patient_id(p):
    base = os.path.splitext(os.path.basename(p))[0]
    parts = base.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else base

# 1) Montgomery + Shenzhen: label from filename suffix _0 / _1
for name in ["montgomery", "shenzhen"]:
    root = PATHS[name]
    if not os.path.isdir(root):
        print("MISSING:", name, root); continue
    for f in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")): continue
        stem = os.path.splitext(os.path.basename(f))[0]
        if   stem.endswith("_1"): lbl = 1
        elif stem.endswith("_0"): lbl = 0
        else: continue
        records.append({"path": f, "label": lbl,
                        "group": f"{name}:{patient_id(f)}", "is_dicom": False})

# 2) TB database: label from folder name (Normal / Tuberculosis)
tb_root = PATHS["tb_database"]
if os.path.isdir(tb_root):
    for f in glob.glob(os.path.join(tb_root, "**", "*.*"), recursive=True):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")): continue
        pl = f.lower()
        if   "/tuberculosis/" in pl: lbl = 1
        elif "/normal/" in pl:       lbl = 0
        else: continue
        # filename is unique per image -> use it as the group id
        gid = os.path.splitext(os.path.basename(f))[0]
        records.append({"path": f, "label": lbl,
                        "group": f"tb_db:{gid}", "is_dicom": False})

# 3) NIRT DICOM: ext is .dicom, label from path, group = patient folder
nirt = PATHS["nirt_dicom"]
if os.path.isdir(nirt):
    for f in glob.glob(os.path.join(nirt, "**", "*"), recursive=True):
        if os.path.isdir(f): continue
        if not f.lower().endswith((".dicom", ".dcm")): continue
        pl = f.lower()
        if   os.sep + "abnormal" + os.sep in pl: lbl = 1
        elif os.sep + "normal"   + os.sep in pl: lbl = 0
        else: continue
        # patient folder is the parent dir name (e.g. intbtr124) -> group key
        pid = os.path.basename(os.path.dirname(f))
        records.append({"path": f, "label": lbl,
                        "group": f"nirt:{pid}", "is_dicom": True})


df = pd.DataFrame(records)
assert len(df) > 0, "No records found - check PATHS in Cell 1"
print(df.groupby(["is_dicom", "label"]).size())
print("\nPer-source counts:")
print(df["group"].str.split(":").str[0].value_counts())
print("\nTotal:", len(df), "| Unlabeled:", df['label'].isna().sum())

def read_dicom(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr - arr.min()
    if arr.max() > 0: arr = arr / arr.max()
    arr = (arr * 255.0).astype(np.uint8)
    img = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
    return np.stack([img]*3, axis=-1)

def read_standard(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.stack([img]*3, axis=-1)

def load_image(path, is_dicom):
    return read_dicom(path) if is_dicom else read_standard(path)

def file_hash(path, is_dicom):
    img = load_image(path, is_dicom)
    return hashlib.md5(img.tobytes()).hexdigest()

df = df.dropna(subset=["label"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

# Content-hash dedup (kills duplicate/near-identical images across datasets)
df["hash"] = [file_hash(p, d) for p, d in zip(df["path"], df["is_dicom"])]
before = len(df)
df = df.drop_duplicates(subset="hash").reset_index(drop=True)
print(f"Removed {before-len(df)} duplicate images, {len(df)} remain")

from sklearn.model_selection import train_test_split

# Build one row per patient group with its dominant label
grp = (df.groupby("group")
         .agg(label=("label", lambda s: int(s.mean() >= 0.5)),
              n=("label", "size"))
         .reset_index())

# Split groups (stratified by group label): 80/10/10
g_train, g_temp = train_test_split(grp, test_size=0.20,
                                   stratify=grp["label"], random_state=SEED)
g_val, g_test   = train_test_split(g_temp, test_size=0.50,
                                   stratify=g_temp["label"], random_state=SEED)

train_df = df[df["group"].isin(g_train["group"])].reset_index(drop=True)
val_df   = df[df["group"].isin(g_val["group"])].reset_index(drop=True)
test_df  = df[df["group"].isin(g_test["group"])].reset_index(drop=True)

# Zero patient overlap
assert not (set(train_df.group) & set(val_df.group))
assert not (set(train_df.group) & set(test_df.group))
assert not (set(val_df.group)   & set(test_df.group))

print("Split sizes -> train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
for nm, fr in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{nm} label %:\n", fr["label"].value_counts(normalize=True).round(3))

def make_ds(frame, training=False):
    paths  = frame["path"].values
    dicoms = frame["is_dicom"].values.astype(np.int32)
    labels = frame["label"].values.astype(np.float32)

    def _load(path, is_dicom, label):
        img = tf.numpy_function(
            lambda p, d: load_image(p.decode(), bool(d)).astype(np.float32),
            [path, is_dicom], tf.float32)
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, dicoms, labels))
    if training:
        ds = ds.shuffle(2048, seed=SEED)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if training:
        aug = tf.keras.Sequential([layers.RandomFlip("horizontal"),
                                   layers.RandomRotation(0.05),
                                   layers.RandomZoom(0.1)])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_ds(train_df, training=True)
val_ds   = make_ds(val_df)
test_ds  = make_ds(test_df)

from sklearn.utils.class_weight import compute_class_weight

# Build model: ImageNet base + dropout + single logit head
def build_model(base_fn):
    base = base_fn(include_top=False, weights="imagenet",
                   input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    x = layers.Dropout(0.3)(base.output)
    out = layers.Dense(1)(x)  # logits, no activation
    return models.Model(base.input, out)

# Class weights to correct train imbalance (~36% TB)
cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                          y=train_df["label"].values)
class_weight = {0: float(cw[0]), 1: float(cw[1])}
print("Class weights:", class_weight)

teacher = build_model(tf.keras.applications.ResNet50)
teacher.compile(optimizer=optimizers.Adam(1e-4),
                loss=losses.BinaryCrossentropy(from_logits=True),
                metrics=[metrics.BinaryAccuracy(threshold=0.0, name="acc"),
                         metrics.AUC(from_logits=True, name="auc")])

teacher.fit(train_ds, validation_data=val_ds, epochs=10,
            class_weight=class_weight,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=3,
                restore_best_weights=True)])

print("\nTeacher test eval:")
teacher.evaluate(test_ds)

class Distiller(models.Model):
    def __init__(self, student, teacher, alpha=0.5, T=3.0):
        super().__init__()
        self.student, self.teacher = student, teacher
        self.alpha, self.T = alpha, T
        self.bce = losses.BinaryCrossentropy(from_logits=True)

    def compile(self, optimizer, **kw):
        super().compile(optimizer=optimizer, **kw)
        self.acc = metrics.BinaryAccuracy(threshold=0.0)
        self.auc = metrics.AUC(from_logits=True, name="auc")

    def _kd_loss(self, t_logits, s_logits):
        t  = tf.sigmoid(t_logits / self.T)
        s  = tf.math.log_sigmoid(s_logits / self.T)
        s0 = tf.math.log_sigmoid(-s_logits / self.T)
        return tf.reduce_mean(-(t * s + (1 - t) * s0)) * (self.T ** 2)

    def train_step(self, data):
        x, y = data
        t_logits = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s_logits = self.student(x, training=True)
            hard = self.bce(y, s_logits)
            soft = self._kd_loss(t_logits, s_logits)
            loss = self.alpha * hard + (1 - self.alpha) * soft
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.acc.update_state(y, tf.sigmoid(s_logits))
        self.auc.update_state(y, tf.sigmoid(s_logits))
        return {"loss": loss, "acc": self.acc.result(), "auc": self.auc.result()}

    def test_step(self, data):
        x, y = data
        s_logits = self.student(x, training=False)
        self.acc.update_state(y, tf.sigmoid(s_logits))
        self.auc.update_state(y, tf.sigmoid(s_logits))
        return {"acc": self.acc.result(), "auc": self.auc.result()}

# Freeze teacher so no test info leaks through distillation
teacher.trainable = False

student = build_model(tf.keras.applications.DenseNet121)
distiller = Distiller(student, teacher, alpha=0.5, T=3.0)
distiller.compile(optimizer=optimizers.Adam(1e-4))

distiller.fit(train_ds, validation_data=val_ds, epochs=15,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_auc", mode="max", patience=4,
                  restore_best_weights=True)])

from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score)

def collect(ds):
    yt, yp = [], []
    for x, y in ds:
        yt.extend(y.numpy())
        yp.extend(tf.sigmoid(student(x, training=False)).numpy().ravel())
    return np.array(yt), np.array(yp)

# 1) Pick best threshold on VALIDATION (never on test)
val_y, val_p = collect(val_ds)
thresholds = np.linspace(0.05, 0.95, 91)
best_t = max(thresholds, key=lambda t: f1_score(val_y, (val_p > t).astype(int)))
print(f"Best threshold (val, max F1): {best_t:.3f}")

# 2) Evaluate on TEST at that threshold
test_y, test_p = collect(test_ds)
test_pred = (test_p > best_t).astype(int)

print("\nStudent Test AUC:", round(roc_auc_score(test_y, test_p), 4))
print("\nAt tuned threshold:")
print(classification_report(test_y, test_pred,
                            target_names=["Normal", "TB"], digits=3))
print("Confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(test_y, test_pred))

# 3) Also show default-0.5 for reference
print("\nAt default 0.5 threshold:")
print(classification_report(test_y, (test_p > 0.5).astype(int),
                            target_names=["Normal", "TB"], digits=3))

# 4) Save student + threshold
student.save("/kaggle/working/tb_student_densenet121.keras")
with open("/kaggle/working/best_threshold.txt", "w") as f:
    f.write(str(best_t))
print("\nSaved model + threshold to /kaggle/working/")

