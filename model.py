import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Carregar o dataset CNN/DailyMail
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Versão reduzida do dataset para treino e teste
praticao_de_treino = (
    dataset["train"].shuffle(seed=42).select(range(500))
)  # 500 exemplos de treino
particao_de_teste = (
    dataset["test"].shuffle(seed=42).select(range(100))
)  # 100 exemplos de validação

# Carregar o modelo T5-small e o tokenizer
model_checkpoint = "t5_small_finetuned"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# Função para tokenizar os dados
def tokenizar(examples):
    # Preparar o texto de entrada e o rótulo (resumo)
    inputs = tokenizer(
        examples["article"], max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["highlights"], max_length=300, truncation=True, padding="max_length"
    )
    inputs["labels"] = labels["input_ids"]
    inputs["decoder_input_ids"] = labels["input_ids"]
    return inputs


# Tokenizar o dataset de treino e teste
particao_treino_tokenizado = praticao_de_treino.map(tokenizar, batched=True)
particao_teste_tokenizado = particao_de_teste.map(tokenizar, batched=True)

# Preparar o dataset de treino e validação em batches para o TensorFlow
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

dataset_treino = particao_treino_tokenizado.to_tf_dataset(
    columns=["input_ids", "attention_mask", "decoder_input_ids"],
    label_cols=["labels"],
    shuffle=True,
    batch_size=10,
    collate_fn=data_collator,
)

dataset_teste = particao_teste_tokenizado.to_tf_dataset(
    columns=["input_ids", "attention_mask", "decoder_input_ids"],
    label_cols=["labels"],
    shuffle=False,
    batch_size=10,
    collate_fn=data_collator,
)

# Compilar o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# Adicionar um callback de EarlyStopping para parar o treinamento se a validação não melhorar
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

# Treinar o modelo
model.fit(
    dataset_treino,
    validation_data=dataset_teste,
    epochs=5,  # Definir número de épocas de acordo com a necessidade
    callbacks=[early_stopping_callback],
)

# Salvar o modelo após o treinamento
model.save_pretrained("t5_small_finetuned")
tokenizer.save_pretrained("t5_small_finetuned")
print("Modelo treinado e salvo com sucesso!")
