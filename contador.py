import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import sqlite3
import json
from datetime import date, datetime
import tkinter as tk # NOVO: Biblioteca para criar janelas visuais

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. FUNÇÃO DA JANELA POP-UP (INTERFACE VISUAL)
# ==========================================
def solicitar_cadastro_gui():
    """Abre uma janela para digitar os dados do novo visitante"""
    janela = tk.Tk()
    janela.title("Novo Visitante Detectado!")
    janela.geometry("350x250")
    
    # Faz a janela saltar para frente da câmera
    janela.attributes('-topmost', True) 
    janela.focus_force()

    tk.Label(janela, text="Pessoa não reconhecida.", fg="red", font=("Arial", 10, "bold")).pack(pady=10)
    
    tk.Label(janela, text="Nome Completo:").pack()
    entrada_nome = tk.Entry(janela, width=30)
    entrada_nome.pack(pady=5)
    
    tk.Label(janela, text="RG / CPF:").pack()
    entrada_doc = tk.Entry(janela, width=30)
    entrada_doc.pack(pady=5)
    
    dados_digitados = {"nome": "Desconhecido", "documento": "Não informado"}
    
    def salvar_e_fechar():
        # Pega o que foi digitado (ou deixa 'Desconhecido' se ficar em branco)
        dados_digitados["nome"] = entrada_nome.get().strip() or "Desconhecido"
        dados_digitados["documento"] = entrada_doc.get().strip() or "Não informado"
        janela.destroy() # Fecha a janela
        
    tk.Button(janela, text="Cadastrar e Liberar Catraca", command=salvar_e_fechar, bg="green", fg="white").pack(pady=20)
    
    # Trava a execução do Python aqui até a pessoa clicar em salvar
    janela.wait_window() 
    return dados_digitados["nome"], dados_digitados["documento"]

# ==========================================
# 2. BANCO DE DADOS ATUALIZADO
# ==========================================
def iniciar_banco():
    conn = sqlite3.connect('visitantes.db')
    cursor = conn.cursor()
    # Adicionamos as colunas NOME e DOCUMENTO
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
    # Agora puxamos o NOME também
    cursor.execute("SELECT id, nome, assinatura_facial FROM visitantes WHERE data_visita = ?", (hoje,))
    linhas = cursor.fetchall()
    
    memoria = []
    for linha in linhas:
        db_id = linha[0]
        nome = linha[1]
        vetor = json.loads(linha[2]) 
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
# 3. INICIALIZAÇÃO DO SISTEMA PRINCIPAL
# ==========================================
print("Conectando ao banco de dados...")
conexao_db = iniciar_banco()
memoria_hoje = carregar_memoria_de_hoje(conexao_db)

print("Iniciando câmeras e IA...")
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

linha_y = 300 
limiar_reconhecimento = 0.55 #tolerancia de aparencia
rastreio_posicoes = {} 

# Dicionário que liga o YOLO diretamente ao NOME da pessoa
mapa_id_yolo_para_dados = {} 

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

            if track_id in rastreio_posicoes:
                cy_anterior = rastreio_posicoes[track_id]
                
                # Se a pessoa cruzar a linha
                if cy_anterior < linha_y and cy >= linha_y:
                    
                    margem = 15
                    recorte_pessoa = frame[max(0, y1-margem):min(frame.shape[0], y2+margem), 
                                           max(0, x1-margem):min(frame.shape[1], x2+margem)]

                    if recorte_pessoa.size > 0:
                        try:
                            # 1. Extrai a biometria
                            embedded = DeepFace.represent(img_path = recorte_pessoa, 
                                                         model_name = 'Facenet', 
                                                         enforce_detection = True, 
                                                         detector_backend = 'mtcnn',
                                                         align = True)
                            
                            nova_assinatura = embedded[0]["embedding"]
                            eh_pessoa_nova = True
                            nome_reconhecido = ""
                            menor_distancia_encontrada = 1.0 #  : Para rastrearmos a nota da IA
                            
                            # 2. Procura no banco de dados de hoje
                            for db_id, nome_banco, assinatura_salva in memoria_hoje:
                                a = np.array(nova_assinatura)
                                b = np.array(assinatura_salva)
                                distancia = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

                                # Salva a menor distância calculada para fins de debug
                                if distancia < menor_distancia_encontrada:
                                    menor_distancia_encontrada = distancia


                                if distancia < limiar_reconhecimento:
                                    eh_pessoa_nova = False
                                    nome_reconhecido = nome_banco
                                    print(f"-> Visitante: {nome_reconhecido} (Já cadastrado)")
                                    break
                            
                            # 3. SE FOR ALGUÉM NOVO, ABRE A TELA DE CADASTRO
                            if eh_pessoa_nova:
                                # AGORA A IA NOS DIZ O MOTIVO DA RECUSA:
                                print(f"*** RECUSADO! Menor distância no banco: {menor_distancia_encontrada:.2f} (Limite é {limiar_reconhecimento}) ***")
                                
                                # Pausa o vídeo e chama a janelinha
                                nome_digitado, doc_digitado = solicitar_cadastro_gui()
                                
                                # Salva no banco com os dados novos
                                id_gerado = salvar_novo_visitante(conexao_db, nova_assinatura, nome_digitado, doc_digitado)
                                
                                # Adiciona na memória para ele não perguntar de novo
                                nome_reconhecido = nome_digitado
                                memoria_hoje.append((id_gerado, nome_reconhecido, nova_assinatura))
                                print(f"*** {nome_reconhecido} cadastrado com sucesso! ***")
                            
                            # 4. Associa o nome à caixinha verde da tela
                            mapa_id_yolo_para_dados[track_id] = nome_reconhecido
                                
                        except Exception as e:
                            print(f"Rosto não nítido. Passe novamente.")

            rastreio_posicoes[track_id] = cy
            
            # --- VISUALIZAÇÃO ---
            # Se temos o nome da pessoa, desenhamos a caixa em AMARELO
            if track_id in mapa_id_yolo_para_dados:
                texto_tela = mapa_id_yolo_para_dados[track_id] # Coloca o nome digitado!
                cor = (0, 255, 255)
            # Se ele ainda não cruzou a linha, deixamos em VERDE com ID genérico
            else:
                texto_tela = f"Aguardando passagem..."
                cor = (0, 255, 0)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame, texto_tela, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    cv2.line(frame, (0, linha_y), (frame.shape[1], linha_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Cadastrados Hoje: {len(memoria_hoje)}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Sistema de Controle de Acesso", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conexao_db.close()