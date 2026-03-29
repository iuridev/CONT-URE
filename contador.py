import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import sqlite3
import json
from datetime import date, datetime
import tkinter as tk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. FUNÇÃO DA JANELA POP-UP (INTERFACE VISUAL)
# ==========================================
def solicitar_cadastro_gui():
    janela = tk.Tk()
    janela.title("Novo Visitante Detetado!")
    janela.geometry("350x250")
    janela.attributes('-topmost', True) 
    janela.focus_force()

    tk.Label(janela, text="Pessoa não reconhecida no radar.", fg="red", font=("Arial", 10, "bold")).pack(pady=10)
    
    tk.Label(janela, text="Nome Completo:").pack()
    entrada_nome = tk.Entry(janela, width=30)
    entrada_nome.pack(pady=5)
    
    tk.Label(janela, text="RG / CPF:").pack()
    entrada_doc = tk.Entry(janela, width=30)
    entrada_doc.pack(pady=5)
    
    dados_digitados = {"nome": "Desconhecido", "documento": "Não informado"}
    
    def salvar_e_fechar():
        dados_digitados["nome"] = entrada_nome.get().strip() or "Desconhecido"
        dados_digitados["documento"] = entrada_doc.get().strip() or "Não informado"
        janela.destroy() 
        
    tk.Button(janela, text="Registar e Liberar Acesso", command=salvar_e_fechar, bg="green", fg="white").pack(pady=20)
    janela.wait_window() 
    return dados_digitados["nome"], dados_digitados["documento"]

# ==========================================
# 2. BANCO DE DADOS
# ==========================================
def iniciar_banco():
    conn = sqlite3.connect('visitantes.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitantes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT,
            documento TEXT,
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
    cursor.execute("SELECT id, nome, assinatura_facial FROM visitantes WHERE data_visita = ?", (hoje,))
    linhas = cursor.fetchall()
    
    memoria = []
    for linha in linhas:
        db_id, nome, vetor_json = linha[0], linha[1], linha[2]
        vetor = json.loads(vetor_json) 
        memoria.append((db_id, nome, vetor))
    return memoria

def salvar_novo_visitante(conn, vetor_facial, nome, documento):
    hoje = date.today().isoformat()
    agora = datetime.now().strftime("%H:%M:%S")
    vetor_json = json.dumps(vetor_facial) 
    
    cursor = conn.cursor()
    cursor.execute("INSERT INTO visitantes (nome, documento, assinatura_facial, data_visita, hora_primeira_entrada) VALUES (?, ?, ?, ?, ?)",
                  (nome, documento, vetor_json, hoje, agora))
    conn.commit()
    return cursor.lastrowid

# ==========================================
# 3. INICIALIZAÇÃO DO SISTEMA
# ==========================================
print("Conectando ao banco de dados...")
conexao_db = iniciar_banco()
memoria_hoje = carregar_memoria_de_hoje(conexao_db)

print("Iniciando câmaras e IA...")
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

# --- CONFIGURAÇÃO DA ZONA DE RADAR ---
linha_radar_y = 150    # Linha Amarela (Início das tentativas)
linha_catraca_y = 350  # Linha Azul (Ponto de decisão final)
limiar_reconhecimento = 0.55 

rastreio_posicoes = {} 
mapa_id_yolo_para_dados = {} 

# Memória de curto prazo para quem está a caminhar dentro do radar
# Formato: { track_id: {"reconhecido": True/False, "nome": "...", "assinatura": [...]} }
cache_radar = {} 

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
            
            cy_anterior = rastreio_posicoes.get(track_id, cy)

            # ==========================================
            # FASE 1: DENTRO DA ZONA DE RADAR (TENTATIVAS)
            # ==========================================
            if linha_radar_y < cy < linha_catraca_y:
                
                # Regista a pessoa no radar se acabou de entrar
                if track_id not in cache_radar:
                    cache_radar[track_id] = {"reconhecido": False, "nome": "", "assinatura": None, "tentativas": 0}
                
                # Se ainda não foi reconhecida, a IA tenta ler o rosto
                if not cache_radar[track_id]["reconhecido"]:
                    cache_radar[track_id]["tentativas"] += 1
                    
                    margem = 15
                    recorte = frame[max(0, y1-margem):min(frame.shape[0], y2+margem), max(0, x1-margem):min(frame.shape[1], x2+margem)]
                    
                    if recorte.size > 0:
                        try:
                            # Tenta extrair a biometria
                            embedded = DeepFace.represent(img_path = recorte, model_name = 'Facenet', 
                                                         enforce_detection = True, detector_backend = 'mtcnn', align = True)
                            nova_assinatura = embedded[0]["embedding"]
                            cache_radar[track_id]["assinatura"] = nova_assinatura # Guarda a melhor foto tirada
                            
                            # Procura no banco de dados
                            for db_id, nome_banco, assinatura_salva in memoria_hoje:
                                a, b = np.array(nova_assinatura), np.array(assinatura_salva)
                                distancia = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

                                if distancia < limiar_reconhecimento:
                                    # SUCESSO! Reconheceu a pessoa no meio do caminho.
                                    cache_radar[track_id]["reconhecido"] = True
                                    cache_radar[track_id]["nome"] = nome_banco
                                    mapa_id_yolo_para_dados[track_id] = nome_banco
                                    print(f"[RADAR] Rosto de {nome_banco} identificado antecipadamente! (Tentativa {cache_radar[track_id]['tentativas']})")
                                    break
                        except Exception:
                            # Rosto não estava bom neste frame. Não faz mal, tenta de novo no próximo!
                            pass

            # ==========================================
            # FASE 2: CRUZOU A CATRACA (DECISÃO FINAL)
            # ==========================================
            if cy_anterior < linha_catraca_y and cy >= linha_catraca_y:
                
                # A pessoa passou pelo radar, vamos ver o veredicto
                if track_id in cache_radar:
                    
                    if cache_radar[track_id]["reconhecido"]:
                        print(f"-> ACESSO LIVRE: {cache_radar[track_id]['nome']} (Validado no Radar)")
                    else:
                        print(f"*** RADAR FALHOU APÓS {cache_radar[track_id]['tentativas']} TENTATIVAS. ABRIR REGISTO. ***")
                        
                        # Usa a última foto tentada no radar (se houver) ou tenta uma nova agora
                        assinatura_final = cache_radar[track_id]["assinatura"]
                        
                        if assinatura_final is None:
                            # Se passou pelo radar inteiro de costas, força uma última extração (pode falhar)
                            print("Aviso: Nenhum rosto captado no radar.")
                            # Aqui poderíamos forçar o utilizador a voltar, mas para a PoC vamos deixar passar
                        
                        # Abre a Janela
                        nome_digitado, doc_digitado = solicitar_cadastro_gui()
                        
                        if assinatura_final is not None:
                            id_gerado = salvar_novo_visitante(conexao_db, assinatura_final, nome_digitado, doc_digitado)
                            memoria_hoje.append((id_gerado, nome_digitado, assinatura_final))
                            
                        mapa_id_yolo_para_dados[track_id] = nome_digitado
                        print(f"*** {nome_digitado} registado com sucesso! ***")
                        
                else:
                    print("Erro: Pessoa 'teletransportou-se' para cima da catraca sem passar pelo radar.")

            rastreio_posicoes[track_id] = cy
            
            # --- VISUALIZAÇÃO ---
            if track_id in mapa_id_yolo_para_dados:
                texto = mapa_id_yolo_para_dados[track_id]
                cor = (0, 255, 255) # Amarelo (Reconhecido)
            elif track_id in cache_radar and not cache_radar[track_id]["reconhecido"]:
                texto = f"A analisar... ({cache_radar[track_id]['tentativas']})"
                cor = (0, 165, 255) # Laranja (A tentar ler o rosto)
            else:
                texto = "YOLO Acompanhando"
                cor = (0, 255, 0) # Verde
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    # Desenha as linhas do Radar na tela
    cv2.line(frame, (0, linha_radar_y), (frame.shape[1], linha_radar_y), (0, 255, 255), 2) # Amarelo: Entrada
    cv2.line(frame, (0, linha_catraca_y), (frame.shape[1], linha_catraca_y), (255, 0, 0), 2) # Azul: Catraca final
    
    cv2.putText(frame, f"Registados Hoje: {len(memoria_hoje)}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Sistema de Acesso com Radar Híbrido", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conexao_db.close()