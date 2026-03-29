import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import sqlite3
import json
from datetime import date, datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CONFIGURAÇÃO DO BANCO DE DADOS (SQLITE)
# ==========================================
def iniciar_banco():
    # Cria o arquivo na mesma pasta do projeto
    conn = sqlite3.connect('visitantes.db')
    cursor = conn.cursor()
    # Cria a tabela se ela não existir
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitantes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assinatura_facial TEXT,
            data_visita DATE,
            hora_primeira_entrada TIME
        )
    ''')
    conn.commit()
    return conn

def carregar_memoria_de_hoje(conn):
    hoje = date.today().isoformat()
    cursor = conn.cursor()
    cursor.execute("SELECT id, assinatura_facial FROM visitantes WHERE data_visita = ?", (hoje,))
    linhas = cursor.fetchall()
    
    memoria = []
    for linha in linhas:
        db_id = linha[0]
        # Transforma o texto do banco de volta em vetor matemático
        vetor = json.loads(linha[1]) 
        memoria.append((db_id, vetor))
    return memoria

def salvar_novo_visitante(conn, vetor_facial):
    hoje = date.today().isoformat()
    agora = datetime.now().strftime("%H:%M:%S")
    # Transforma o vetor matemático em texto para salvar no SQLite
    vetor_json = json.dumps(vetor_facial) 
    
    cursor = conn.cursor()
    cursor.execute("INSERT INTO visitantes (assinatura_facial, data_visita, hora_primeira_entrada) VALUES (?, ?, ?)",
                  (vetor_json, hoje, agora))
    conn.commit()
    return cursor.lastrowid # Retorna o ID gerado pelo banco (1, 2, 3...)

# ==========================================
# 2. INICIALIZAÇÃO DO SISTEMA
# ==========================================
print("Conectando ao banco de dados local...")
conexao_db = iniciar_banco()
memoria_hoje = carregar_memoria_de_hoje(conexao_db)
total_visitantes_hoje = len(memoria_hoje)

print(f"Banco carregado! Visitantes já registrados hoje: {total_visitantes_hoje}")
print("Iniciando câmeras e IA...")

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

linha_y = 300 
limiar_reconhecimento = 0.45 
rastreio_posicoes = {} 

# Dicionário para trocar o ID do YOLO pelo ID do Banco na tela
mapa_id_yolo_para_db = {} 

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        break

    resultados = model.track(frame, persist=True, classes=[0], verbose=False)

    if resultados[0].boxes.id is not None:
        boxes = resultados[0].boxes.xyxy.cpu()
        track_ids = resultados[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # LÓGICA DE CRUZAMENTO DE LINHA
            if track_id in rastreio_posicoes:
                cy_anterior = rastreio_posicoes[track_id]
                
                # Se cruzou a linha
                if cy_anterior < linha_y and cy >= linha_y:
                    
                    margem = 15
                    recorte_pessoa = frame[max(0, y1-margem):min(frame.shape[0], y2+margem), 
                                           max(0, x1-margem):min(frame.shape[1], x2+margem)]

                    if recorte_pessoa.size > 0:
                        try:
                            # Extrai a assinatura com MTCNN
                            embedded = DeepFace.represent(img_path = recorte_pessoa, 
                                                         model_name = 'Facenet', 
                                                         enforce_detection = True, 
                                                         detector_backend = 'mtcnn',
                                                         align = True)
                            
                            nova_assinatura = embedded[0]["embedding"]
                            eh_pessoa_nova = True
                            id_banco = None
                            
                            # Compara com a memória carregada do banco de dados
                            for db_id, assinatura_salva in memoria_hoje:
                                a = np.array(nova_assinatura)
                                b = np.array(assinatura_salva)
                                distancia = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

                                if distancia < limiar_reconhecimento:
                                    eh_pessoa_nova = False
                                    id_banco = db_id
                                    print(f"-> Rosto conhecido! É o Visitante {id_banco} (Similaridade: {distancia:.2f})")
                                    break
                            
                            # Se a IA não achou ninguem parecido no banco
                            if eh_pessoa_nova:
                                # Salva no SQLite e pega o ID oficial
                                id_banco = salvar_novo_visitante(conexao_db, nova_assinatura)
                                
                                # Atualiza a memória RAM para as próximas passadas
                                memoria_hoje.append((id_banco, nova_assinatura))
                                total_visitantes_hoje = len(memoria_hoje)
                                print(f"*** NOVO VISITANTE SALVO *** ID Banco: {id_banco}")
                            
                            # Associa o ID volátil do YOLO com o ID permanente do Banco
                            mapa_id_yolo_para_db[track_id] = id_banco
                                
                        except Exception as e:
                            print(f"ID {track_id} cruzou, mas o rosto não estava nítido para o banco.")

            rastreio_posicoes[track_id] = cy
            
            # VISUALIZAÇÃO NA TELA
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Se já reconhecemos a pessoa, mostra "Visitante X", senão mostra o tracking do YOLO
            if track_id in mapa_id_yolo_para_db:
                texto_tela = f"Visitante {mapa_id_yolo_para_db[track_id]}"
                cor_texto = (0, 255, 255) # Amarelo para pessoas registradas
            else:
                texto_tela = f"YOLO: {track_id}"
                cor_texto = (0, 255, 0) # Verde para tracking inicial
                
            cv2.putText(frame, texto_tela, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 2)

    cv2.line(frame, (0, linha_y), (frame.shape[1], linha_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Unicos Hoje: {total_visitantes_hoje}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Contador Inteligente c/ SQLite", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpeza
cap.release()
cv2.destroyAllWindows()
conexao_db.close()