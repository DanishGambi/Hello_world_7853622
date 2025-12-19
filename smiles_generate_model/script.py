# Установка библиотек:

# pip install -r requirements.txt # в терминал - для установки библиотек


#########################################################

# Необходимые функции (архитектура нейронки + загрузка модели и токенизатора):

#########################################################

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from rdkit import Chem
import pandas as pd
import numpy as np

# Параметры (должны совпадать с обучением!)
input_dim = 10
embed_dim = 128
hidden_dim = 256
vocab_size = 410  # Загрузим точное значение из токенизатора позже
max_len = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # !!! если не планируется использовать gpu, то меняем на device = torch.device("cpu") 

# --- ОПРЕДЕЛЕНИЕ МОДЕЛИ (точно как при обучении) ---
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return torch.tanh(self.fc(x))

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_proj = nn.Linear(hidden_dim, hidden_dim)  # Проекция для conditioning
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len

    def forward(self, encoder_out, targets, teacher_forcing=True):
        batch_size, seq_len = targets.size()
        # Стартовый input: embedding <start>
        inputs = self.embedding(targets[:, 0]).unsqueeze(1)  # batch x 1 x embed
        # Encoder output повторяется по seq_len
        encoder_repeated = self.encoder_proj(encoder_out).unsqueeze(1).repeat(1, seq_len, 1)  # batch x seq x hidden
        outputs = []
        hidden = None
        for t in range(1, seq_len):  # Начинаем с t=1
            encoder_step = encoder_repeated[:, t, :].unsqueeze(1)  # batch x 1 x hidden
            combined = torch.cat([inputs, encoder_step], dim=-1)  # batch x 1 x (embed + hidden)
            out, hidden = self.lstm(combined, hidden)
            logit = self.fc(out.squeeze(1))  # batch x vocab
            outputs.append(logit)
            # Teacher forcing
            if teacher_forcing and t < seq_len:
                next_input = self.embedding(targets[:, t])
            else:
                next_input = self.embedding(torch.argmax(logit, dim=1))  # greedy
            inputs = next_input.unsqueeze(1)  # batch x 1 x embed
        return torch.stack(outputs, dim=1)  # batch x (seq-1) x vocab

class MoleculeGenerator(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, vocab_size, max_len):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(embed_dim, hidden_dim, vocab_size, max_len)

    def forward(self, features, targets, teacher_forcing=True):
        encoder_out = self.encoder(features)
        return self.decoder(encoder_out, targets, teacher_forcing)

# --- ФУНКЦИЯ ГЕНЕРАЦИИ SMILES ---
def generate_smiles(model, features, tokenizer, max_len=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        features = features.unsqueeze(0).to(device) if features.dim() == 1 else features.to(device)  # batch=1
        encoder_out = model.encoder(features)  # 1 x hidden
        generated_ids = [tokenizer.token_to_id("<start>")]
        input_seq = torch.tensor([[tokenizer.token_to_id("<start>")]], dtype=torch.long).to(device)
        hidden = None
        for _ in range(max_len - 1):  # -1 для start
            embedded = model.decoder.embedding(input_seq)  # 1 x 1 x embed
            encoder_proj = model.decoder.encoder_proj(encoder_out).unsqueeze(1)  # 1 x 1 x hidden
            combined = torch.cat([embedded, encoder_proj], dim=-1)  # 1 x 1 x (embed + hidden)
            out, hidden = model.decoder.lstm(combined, hidden)
            logit = model.decoder.fc(out.squeeze(1))  # 1 x vocab
            # Temperature-based sampling
            if temperature > 0:
                probs = torch.softmax(logit / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(logit, dim=1).item()
            generated_ids.append(next_token)
            if next_token == tokenizer.token_to_id("<end>"):
                break
            input_seq = torch.tensor([[next_token]], dtype=torch.long).to(device)
        # Decode ids в строку: убираем <start> и <end>, decode to SMILES
        generated_smiles_ids = generated_ids[1:]  # Убираем <start>
        if generated_smiles_ids and generated_smiles_ids[-1] == tokenizer.token_to_id("<end>"):
            generated_smiles_ids = generated_smiles_ids[:-1]  # Убираем <end>
        generated_smiles = tokenizer.decode(generated_smiles_ids, skip_special_tokens=True)  # BPE decode в readable SMILES
        return generated_smiles

# --- ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА ---
# print("Загружаем токенизатор...")
tokenizer = Tokenizer.from_file("smiles_bpe.json")
actual_vocab_size = tokenizer.get_vocab_size()
# print(f"Фактический размер словаря: {actual_vocab_size}")

# print("Создаём модель...")
model = MoleculeGenerator(input_dim, embed_dim, hidden_dim, actual_vocab_size, max_len).to(device)

# print("Загружаем веса модели...")
model.load_state_dict(torch.load("best_molecule_generator.pth", map_location=device, weights_only=True))
model.eval()

#########################################################

# Данные ниже задаёт пользователь (+ проверки на адекватность вводимых данных):

#########################################################

MW = 336.775
LogP = 2.29702
TPSA = 91.68
HBD = 3
HBA = 6
RB = 6
Atoms = 23
HeavyAtoms = 23
AromaticAtoms = 12
ChargedAtoms = 0


assert MW >= 18 and MW <= 1500, 'Параметр MW вне диапазона MW > 18 and MW < 1500'
assert LogP >= -10 and LogP <= 10, 'Параметр LogP вне диапазона LogP > -10 and LogP < 10'
assert TPSA >= 0 and TPSA <= 400, 'Параметр TPSA вне диапазона TPSA > 0 and TPSA < 400'
assert HBD >= 0 and HBD <= 15, 'Параметр HBD вне диапазона HBD > 0 and HBD < 15'
assert HBA >= 0 and HBA <= 25, 'Параметр HBA вне диапазона HBA > 0 and HBA < 25'
assert RB >= 0 and RB <= 50, 'Параметр RB вне диапазона RB > 0 and RB < 50'
assert Atoms >= 3 and Atoms <= 200, 'Параметр Atoms вне диапазона Atoms > 3 and Atoms < 200'
assert HeavyAtoms >= 1 and HeavyAtoms <= 150, 'Параметр HeavyAtoms вне диапазона HeavyAtoms > 1 and HeavyAtoms < 150'
assert AromaticAtoms >= 0 and AromaticAtoms <= 80, 'Параметр AromaticAtoms вне диапазона AromaticAtoms > 0 and AromaticAtoms < 80'
assert ChargedAtoms >= 0 and ChargedAtoms <= 20, 'Параметр ChargedAtoms вне диапазона ChargedAtoms > 0 and ChargedAtoms < 20'

assert type(MW) == float or type(MW) == int , 'Тип данных MW должен быть целым или дробным числом.'
assert type(LogP) == float or type(LogP) == int, 'Тип данных LogP должен быть целым или дробным числом.'
assert type(TPSA) == float or type(TPSA) == int, 'Тип данных TPSA должен быть целым или дробным числом.'
assert type(HBD) == int, 'Тип данных HBD должен быть целым числом.'
assert type(HBA) == int, 'Тип данных HBA должен быть целым числом.'
assert type(RB) == int, 'Тип данных RB должен быть целым числом.'
assert type(Atoms) == int, 'Тип данных Atoms должен быть целым числом.'
assert type(HeavyAtoms) == int, 'Тип данных HeavyAtoms должен быть целым числом.'
assert type(AromaticAtoms) == int, 'Тип данных AromaticAtoms должен быть целым числом.'
assert type(ChargedAtoms) == int, 'Тип данных ChargedAtoms должен быть целым числом.'

#########################################################

# Далее генерация smiles и анализ результатов:

#########################################################

from rdkit.Chem import Descriptors
import numpy as np
from sklearn.preprocessing import StandardScaler

N = 1000 # количество генерируемых молекул (если ставить больше - получим лучше качество, но медленнее генерация, если ставить меньше - хуже качество, но быстрее)


features = [MW, LogP, TPSA, HBD, HBA, RB, Atoms, HeavyAtoms, AromaticAtoms, ChargedAtoms]
example_features = torch.tensor(features, dtype=torch.float32)

# 🔇 ОТКЛЮЧАЕМ ВСЕ ПРЕДУПРЕЖДЕНИЯ RDKit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# === ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: надёжная проверка валидности ===
def validate_and_canonicalize(smiles):
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        return mol
    except:
        return None

# === ОСНОВНОЙ КОД ===
print("Генерируем SMILES...")

valid_entries = []

for i in range(N):
    smiles = generate_smiles(model, example_features, tokenizer, temperature=0.8)
    mol = validate_and_canonicalize(smiles)

    if mol is not None:
        props = np.array([
            Chem.Descriptors.MolWt(mol),
            Chem.Descriptors.MolLogP(mol),
            Chem.Descriptors.TPSA(mol),
            Chem.Descriptors.NumHDonors(mol),
            Chem.Descriptors.NumHAcceptors(mol),
            Chem.Descriptors.NumRotatableBonds(mol),
            mol.GetNumAtoms(),
            mol.GetNumHeavyAtoms(),
            sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
            sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
        ], dtype=np.float64)
        valid_entries.append((smiles, mol, props))

if not valid_entries:
    print("❌ Не удалось сгенерировать валидную молекулу.")
else:
    all_props = np.array([entry[2] for entry in valid_entries])
    scaler = StandardScaler()
    all_props_scaled = scaler.fit_transform(all_props)

    user_props = np.array([
        MW, LogP, TPSA, HBD, HBA, RB,
        Atoms, HeavyAtoms, AromaticAtoms, ChargedAtoms
    ], dtype=np.float64).reshape(1, -1)
    user_props_scaled = scaler.transform(user_props)
    user_vec = user_props_scaled.flatten()

    distances = np.linalg.norm(all_props_scaled - user_vec, axis=1)
    best_idx = int(np.argmin(distances))
    best_smiles, best_mol, best_props = valid_entries[best_idx]
    canonical = Chem.MolToSmiles(best_mol, canonical=True)

    MW_gen, LogP_gen, TPSA_gen, HBD_gen, HBA_gen, RB_gen, \
    Atoms_gen, HeavyAtoms_gen, AromaticAtoms_gen, ChargedAtoms_gen = best_props

    def similarity_percent(gen, target):
        if abs(target) < 1e-8:
            return 100.0 if abs(gen) < 1e-8 else 0.0
        return max(0.0, 100.0 - 100.0 * abs(gen - target) / abs(target))

    similarities = [
        similarity_percent(MW_gen, MW),
        similarity_percent(LogP_gen, LogP),
        similarity_percent(TPSA_gen, TPSA),
        similarity_percent(HBD_gen, HBD),
        similarity_percent(HBA_gen, HBA),
        similarity_percent(RB_gen, RB),
        similarity_percent(Atoms_gen, Atoms),
        similarity_percent(HeavyAtoms_gen, HeavyAtoms),
        similarity_percent(AromaticAtoms_gen, AromaticAtoms),
        similarity_percent(ChargedAtoms_gen, ChargedAtoms)
    ]
    avg_similarity = np.mean(similarities)

    print("\n✅ Сгенерирована валидная молекула!")
    print(f"Сгенерированный SMILES: {best_smiles}")
    print(f"Канонический SMILES: {canonical}")
    print(f"   Generated molecule properties: "
          f"MW={MW_gen:.2f}, LogP={LogP_gen:.2f}, TPSA={TPSA_gen:.2f}, "
          f"HBD={HBD_gen:.2f}, HBA={HBA_gen:.2f}, RB={RB_gen:.2f}, "
          f"Atoms={Atoms_gen:.2f}, HeavyAtoms={HeavyAtoms_gen:.2f}, "
          f"AromaticAtoms={AromaticAtoms_gen:.2f}, ChargedAtoms={ChargedAtoms_gen:.2f}")
    print(f"   Your molecule properties: "
          f"MW={MW:.2f}, LogP={LogP:.2f}, TPSA={TPSA:.2f}, "
          f"HBD={HBD:.2f}, HBA={HBA:.2f}, RB={RB:.2f}, "
          f"Atoms={Atoms:.2f}, HeavyAtoms={HeavyAtoms:.2f}, "
          f"AromaticAtoms={AromaticAtoms:.2f}, ChargedAtoms={ChargedAtoms:.2f}")
    print(f"   Схожесть параметров: {avg_similarity:.2f}%")