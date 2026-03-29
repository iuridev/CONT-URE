import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import sqlite3
import json
from datetime import date, datetime
import tkinter as tk
import threading 
import queue     

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. INTERFACE GRÁFICA (ECRÃ DE REGISTO)
# ==========================================
def solicitar_cadastro_gui():
    janela = tk.Tk()
    janela.title("Novo Visitante")
    janela.geometry("350x250")
    janela.attributes('-topmost', True) 
    janela.focus_force()

    tk.Label(janela, text="Biometria não encontrada.", fg="red", font=("Arial", 10, "bold")).pack(pady=10)
    tk.Label(janela, text="Nome Completo:").pack()
    entrada_nome = tk.Entry(janela, width=30)
    entrada_nome.pack(pady=5)
    tk.Label(janela, text="RG / CPF:").pack()
    entrada_doc = tk.Entry(janela, width=30)
    entrada_doc.pack(pady=5)
    
    dados = {"nome": "Desconhecido", "documento": "Não informado"}
    
    def salvar():
        dados["nome"] = entrada_nome.get().strip() or "Desconhecido"
        dados["documento"] = entrada_doc.get().strip() or "Não informado"
        janela.destroy() 
        
    tk.Button(janela, text="Registar e Liberar", command=salvar, bg="green", fg="white").pack(pady=20)
    janela.wait_window() 
    return dados["nome"], dados["documento"]

# ==========================================
# 2. BANCO DE DADOS
# ==========================================
def iniciar_banco():
    conn = sqlite3.connect('controle_acesso.db') 
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS pessoas (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT, documento TEXT, assinatura_facial TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS visitas (id INTEGER PRIMARY KEY AUTOINCREMENT, pessoa_id INTEGER, data_visita DATE, hora_visita TIME, FOREIGN KEY(pessoa_id) REFERENCES pessoas(id))')
    conn.commit()
    return conn

def carregar_todas_pessoas(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome, assinatura_facial FROM pessoas")
    return [(linha[0], linha[1], json.loads(linha[2])) for linha in cursor.fetchall()]

def salvar_nova_pessoa(conn, vetor, nome, doc):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO pessoas (nome, documento, assinatura_facial) VALUES (?, ?, ?)", (nome, doc, json.dumps(vetor)))
    conn.commit()
    return cursor.lastrowid

def registrar_visita_hoje(conn, pessoa_id):
    hoje = date.today().isoformat()
    agora = datetime.now().strftime("%H:%M:%S")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM visitas WHERE pessoa_id = ? AND data_visita = ?", (pessoa_id, hoje))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO visitas (pessoa_id, data_visita, hora_visita) VALUES (?, ?, ?)", (pessoa_id, hoje, agora))
        conn.commit()
        return True # Retorna True se for uma visita nova hoje
    return False # Retorna False se a pessoa já tinha entrado hoje

# NOVO: Função para contar quantos visitantes únicos tivemos hoje
def contar_visitantes_hoje(conn):
    hoje = date.today().isoformat()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT pessoa_id) FROM visitas WHERE data_visita = ?", (hoje,))
    return cursor.fetchone()[0]

def buscar_ultimas_visitas(conn, pessoa_id):
    hoje = date.today().isoformat()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT data_visita FROM visitas WHERE pessoa_id = ? AND data_visita != ? ORDER BY data_visita DESC LIMIT 6', (pessoa_id, hoje))
    return [datetime.strptime(linha[0], "%Y-%m-%d").strftime("%d/%m/%Y") for linha in cursor.fetchall()]

# ==========================================
# 3. HUD TRANSLÚCIDO
# ==========================================
def desenhar_painel_historico(frame, x1_box, y1_box, x2_box, nome, visitas):
    largura_painel = 180
    altura_linha = 20
    altura_total = 35 + (len(visitas) * altura_linha) if visitas else 55
    x_painel = x2_box + 10 
    if x_painel + largura_painel > frame.shape[1]: x_painel = max(0, x1_box - largura_painel - 10) 
    y_painel = max(0, y1_box)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_painel, y_painel), (x_painel + largura_painel, y_painel + altura_total), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, "Ultimos acessos:", (x_painel + 10, y_painel + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if not visitas:
        cv2.putText(frame, "-> Primeira visita!", (x_painel + 10, y_painel + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        for i, dt in enumerate(visitas):
            cv2.putText(frame, f"- {dt}", (x_painel + 10, y_painel + 45 + (i * altura_linha)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

# ==========================================
# 4. THREAD DA INTELIGÊNCIA ARTIFICIAL
# ==========================================
fila_fotos = queue.Queue()
limiar_reconhecimento = 0.55 

def trabalhador_ia():
    while True:
        tarefa = fila_fotos.get()
        if tarefa is None: break 
        
        track_id, recorte_rosto = tarefa
        
        try:
            embedded = DeepFace.represent(img_path=recorte_rosto, model_name='Facenet', enforce_detection=True, detector_backend='mtcnn', align=True)
            nova_assinatura = embedded[0]["embedding"]
            
            reconhecido = False
            for db_id, nome_banco, assinatura_salva in memoria_pessoas:
                a, b = np.array(nova_assinatura), np.array(assinatura_salva)
                dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

                if dist < limiar_reconhecimento:
                    estado_rostos[track_id]["estado"] = "reconhecido"
                    estado_rostos[track_id]["db_id"] = db_id
                    estado_rostos[track_id]["nome"] = nome_banco
                    estado_rostos[track_id]["assinatura"] = nova_assinatura
                    reconhecido = True
                    break
            
            if not reconhecido:
                estado_rostos[track_id]["estado"] = "desconhecido"
                estado_rostos[track_id]["assinatura"] = nova_assinatura
                
        except Exception:
            estado_rostos[track_id]["estado"] = "aguardando"
            
        fila_fotos.task_done()

thread_ia = threading.Thread(target=trabalhador_ia, daemon=True)
thread_ia.start()

# ==========================================
# 5. LOOP PRINCIPAL (VISÃO POR PRESENÇA)
# ==========================================
print("A ligar ao banco de dados...")
conexao_db = iniciar_banco()
memoria_pessoas = carregar_todas_pessoas(conexao_db)

# NOVO: Puxa do banco de dados o valor inicial ao ligar o sistema
total_visitantes_hoje = contar_visitantes_hoje(conexao_db) 

print("A iniciar sistema Free Flow...")
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

estado_rostos = {} 
mapa_dados_tela = {} 

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso: break

    resultados = model.track(frame, persist=True, classes=[0], verbose=False)

    if resultados[0].boxes.id is not None:
        boxes = resultados[0].boxes.xyxy.cpu()
        track_ids = resultados[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            if track_id not in estado_rostos:
                estado_rostos[track_id] = {"estado": "aguardando", "db_id": None, "nome": "", "assinatura": None, "visita_registada": False}
            
            estado_atual = estado_rostos[track_id]["estado"]
            
            if estado_atual == "aguardando":
                margem = 15
                recorte = frame[max(0, y1-margem):min(frame.shape[0], y2+margem), max(0, x1-margem):min(frame.shape[1], x2+margem)]
                
                if recorte.size > 0:
                    estado_rostos[track_id]["estado"] = "processando" 
                    fila_fotos.put((track_id, recorte.copy())) 

            elif estado_atual == "reconhecido":
                if not estado_rostos[track_id]["visita_registada"]:
                    db_id = estado_rostos[track_id]["db_id"]
                    
                    # NOVO: Se for a primeira vez que a pessoa entra hoje, aumenta o contador
                    nova_visita = registrar_visita_hoje(conexao_db, db_id)
                    if nova_visita:
                        total_visitantes_hoje += 1
                        
                    estado_rostos[track_id]["visita_registada"] = True
                    historico = buscar_ultimas_visitas(conexao_db, db_id)
                    mapa_dados_tela[track_id] = {"nome": estado_rostos[track_id]["nome"], "visitas": historico}

            elif estado_atual == "desconhecido":
                cv2.putText(frame, "A ABRIR REGISTO...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Sistema de Acesso Livre", frame) 
                cv2.waitKey(1) 
                
                assinatura_final = estado_rostos[track_id]["assinatura"]
                nome_digitado, doc_digitado = solicitar_cadastro_gui()
                
                if assinatura_final is not None:
                    novo_id = salvar_nova_pessoa(conexao_db, assinatura_final, nome_digitado, doc_digitado)
                    registrar_visita_hoje(conexao_db, novo_id)
                    
                    # NOVO: Sendo uma pessoa nova acabada de registar, conta como visita
                    total_visitantes_hoje += 1
                    
                    memoria_pessoas.append((novo_id, nome_digitado, assinatura_final))
                    
                    estado_rostos[track_id]["estado"] = "reconhecido"
                    estado_rostos[track_id]["db_id"] = novo_id
                    estado_rostos[track_id]["nome"] = nome_digitado
                    estado_rostos[track_id]["visita_registada"] = True
                    mapa_dados_tela[track_id] = {"nome": nome_digitado, "visitas": []} 

            # --- DESENHO NO ECRÃ ---
            if track_id in mapa_dados_tela:
                dados = mapa_dados_tela[track_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, dados["nome"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                desenhar_painel_historico(frame, x1, y1, x2, dados["nome"], dados["visitas"])
                
            elif estado_atual == "processando":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, "A pensar...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # NOVO: Desenha o contador de visitantes únicos no canto superior esquerdo do vídeo
    cv2.putText(frame, f"Visitantes Hoje: {total_visitantes_hoje}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Sistema de Acesso Livre", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpeza e encerramento
fila_fotos.put(None) 
cap.release()
cv2.destroyAllWindows()
conexao_db.close()