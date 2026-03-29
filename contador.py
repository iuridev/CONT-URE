"""
SISTEMA DE ACESSO LIVRE (FREE FLOW) COM FILA DE ATENDIMENTO
Arquitetura: Visão Computacional (OpenCV + YOLO) + Biometria (DeepFace) + GUI (Tkinter) + Concorrência (Threads)
"""

import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import sqlite3
import json
from datetime import date, datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading 
import queue     

# Desativa mensagens de log e avisos vermelhos do TensorFlow no terminal para manter a consola limpa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ======================================================================
# ⚙️ ÁREA DE CONFIGURAÇÕES (Pode alterar aqui sem quebrar o código)
# ======================================================================
CONFIG = {
    "CAMERA_ID": 0,                   # 0 = Webcam padrão. Troque para 1, 2, ou link RTSP se usar câmaras externas.
    "MODELO_YOLO": 'yolov8n.pt',      # 'yolov8n.pt' (Nano) é mais rápido. 'yolov8s.pt' (Small) é mais preciso, mas pesado.
    "TOLERANCIA_FACIAL": 0.55,        # Limite da IA. Quanto MENOR, mais rigoroso (ex: 0.40). Quanto MAIOR, mais permissivo.
    "FPS_TELA": 15,                   # Velocidade de atualização do vídeo no Tkinter (em milissegundos). 15ms ~= 60fps.
    "MARGEM_ROSTO": 15                # Pixels extras ao redor da caixa do rosto antes de enviar para a IA (ajuda na precisão).
}


# ======================================================================
# 💾 MÓDULO DE BANCO DE DADOS (SQLite)
# ======================================================================
def iniciar_banco():
    """Cria o ficheiro do banco de dados e as tabelas se não existirem."""
    conn = sqlite3.connect('controle_acesso.db') 
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS pessoas (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT, documento TEXT, assinatura_facial TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS visitas (id INTEGER PRIMARY KEY AUTOINCREMENT, pessoa_id INTEGER, data_visita DATE, hora_visita TIME, FOREIGN KEY(pessoa_id) REFERENCES pessoas(id))')
    conn.commit()
    return conn

def carregar_todas_pessoas(conn):
    """Carrega toda a biometria para a memória RAM (lista) ao ligar o sistema para reconhecimento em tempo real."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome, assinatura_facial FROM pessoas")
    # Converte o texto JSON de volta para o vetor matemático (lista de números)
    return [(linha[0], linha[1], json.loads(linha[2])) for linha in cursor.fetchall()]

def salvar_nova_pessoa(conn, vetor, nome, doc):
    """Regista um visitante inédito e devolve o ID gerado pelo banco."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO pessoas (nome, documento, assinatura_facial) VALUES (?, ?, ?)", (nome, doc, json.dumps(vetor)))
    conn.commit()
    return cursor.lastrowid

def buscar_pessoa_por_documento(conn, documento):
    """Procura se a pessoa já existe pelo RG/CPF (Usado para tratar falsas rejeições)."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome FROM pessoas WHERE documento = ?", (documento,))
    return cursor.fetchone() 

def atualizar_assinatura_pessoa(conn, pessoa_id, nova_assinatura):
    """Sobrescreve a foto antiga pela nova. É assim que o sistema "aprende" se a pessoa mudar de visual."""
    cursor = conn.cursor()
    cursor.execute("UPDATE pessoas SET assinatura_facial = ? WHERE id = ?", (json.dumps(nova_assinatura), pessoa_id))
    conn.commit()

def registrar_visita_hoje(conn, pessoa_id):
    """Garante que a pessoa só conte como '1 visitante' por dia, não importa quantas vezes passe na câmara."""
    hoje = date.today().isoformat()
    agora = datetime.now().strftime("%H:%M:%S")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM visitas WHERE pessoa_id = ? AND data_visita = ?", (pessoa_id, hoje))
    
    if not cursor.fetchone(): # Se não encontrou registo hoje, insere.
        cursor.execute("INSERT INTO visitas (pessoa_id, data_visita, hora_visita) VALUES (?, ?, ?)", (pessoa_id, hoje, agora))
        conn.commit()
        return True # Retorna True para sabermos que a contagem do ecrã deve subir
    return False

def contar_visitantes_hoje(conn):
    """Busca o número total de visitantes únicos do dia para exibir no canto da tela."""
    hoje = date.today().isoformat()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT pessoa_id) FROM visitas WHERE data_visita = ?", (hoje,))
    return cursor.fetchone()[0]


# ======================================================================
# 🧠 MÓDULO DE INTELIGÊNCIA ARTIFICIAL (Roda em 2º Plano)
# ======================================================================
# A fila serve como um "tubo" entre a câmara (rápida) e a IA (lenta).
fila_fotos_ia = queue.Queue()

# Dicionários Globais de Estado:
estado_rostos = {}        # Mantém o status de quem está na tela: aguardando, processando, reconhecido, na_fila.
cadastros_pendentes = {}  # Guarda a foto e a biometria de quem precisa fazer o registo manual.
mapa_dados_tela = {}      # Liga o ID do YOLO ao Nome que será desenhado no vídeo.

def trabalhador_ia():
    """
    Função Worker (Thread). Fica invisível a tentar processar fotos.
    Como roda em paralelo, o DeepFace não trava o vídeo principal do OpenCV.
    """
    while True:
        tarefa = fila_fotos_ia.get() # Pega a próxima foto da fila
        if tarefa is None: break     # Sinal de encerramento do sistema
        
        track_id, recorte_rosto = tarefa
        
        try:
            # 1. Extrai a biometria da foto recebida (O trabalho pesado)
            embedded = DeepFace.represent(img_path=recorte_rosto, model_name='Facenet', enforce_detection=True, detector_backend='mtcnn', align=True)
            nova_assinatura = embedded[0]["embedding"]
            
            reconhecido = False
            # 2. Compara com todas as pessoas da memória RAM
            for db_id, nome_banco, assinatura_salva in memoria_pessoas:
                a, b = np.array(nova_assinatura), np.array(assinatura_salva)
                # Cálculo de Distância por Cosseno
                dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

                if dist < CONFIG["TOLERANCIA_FACIAL"]:
                    # SUCESSO! Pessoa reconhecida.
                    estado_rostos[track_id]["estado"] = "reconhecido"
                    estado_rostos[track_id]["db_id"] = db_id
                    estado_rostos[track_id]["nome"] = nome_banco
                    estado_rostos[track_id]["assinatura"] = nova_assinatura
                    reconhecido = True
                    break
            
            if not reconhecido:
                # FALHOU. Desconhecido. Envia os dados para a barra lateral direita do Tkinter.
                estado_rostos[track_id]["estado"] = "na_fila"
                cadastros_pendentes[track_id] = {
                    "assinatura": nova_assinatura,
                    "foto": recorte_rosto.copy(),
                    "widget": None # Será preenchido quando o Tkinter desenhar o "cartão"
                }
                
        except Exception:
            # Erro na leitura (ex: rosto muito borrado). Devolve o status para a câmara tentar enviar outra foto.
            estado_rostos[track_id]["estado"] = "aguardando"
            
        fila_fotos_ia.task_done()


# ======================================================================
# 🖥️ MÓDULO DE INTERFACE GRÁFICA E CÂMARA (Centro de Comando)
# ======================================================================
class CentroDeComandoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Centro de Comando - Acesso Livre")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.total_visitantes_hoje = contar_visitantes_hoje(conexao_db)

        # Divisão de Layout: Frame Esquerdo (Vídeo) e Frame Direito (Fila)
        self.frame_esq = tk.Frame(self.root, bg="#000000", width=800, height=600)
        self.frame_esq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_dir = tk.Frame(self.root, bg="#2d2d2d", width=350)
        self.frame_dir.pack(side=tk.RIGHT, fill=tk.Y)
        self.frame_dir.pack_propagate(False) # Força a largura fixa de 350px

        tk.Label(self.frame_dir, text="Fila de Registo", font=("Arial", 16, "bold"), bg="#2d2d2d", fg="white").pack(pady=10)

        # Widget de imagem onde o frame do OpenCV será colado
        self.lbl_video = tk.Label(self.frame_esq, bg="black")
        self.lbl_video.pack(expand=True, fill=tk.BOTH)

        # Inicializa Modelos
        self.model = YOLO(CONFIG["MODELO_YOLO"])
        self.cap = cv2.VideoCapture(CONFIG["CAMERA_ID"])
        
        # Inicia a Thread da IA
        self.thread_ia = threading.Thread(target=trabalhador_ia, daemon=True)
        self.thread_ia.start()

        # Inicia os Loops do Tkinter
        self.atualizar_video()
        self.atualizar_fila()

    def atualizar_video(self):
        """Lê a câmara, processa o tracking do YOLO e desenha as caixas na tela."""
        sucesso, frame = self.cap.read()
        if sucesso:
            # Tracking persitente para manter o ID da pessoa enquanto ela caminha
            resultados = self.model.track(frame, persist=True, classes=[0], verbose=False)

            if resultados[0].boxes.id is not None:
                boxes = resultados[0].boxes.xyxy.cpu()
                track_ids = resultados[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Se for um ID novo (pessoa nova no ecrã), regista no dicionário de estados
                    if track_id not in estado_rostos:
                        estado_rostos[track_id] = {"estado": "aguardando", "db_id": None, "nome": "", "assinatura": None, "visita_registada": False}
                    
                    estado_atual = estado_rostos[track_id]["estado"]
                    
                    # MÁQUINA DE ESTADOS:
                    # 1. Tenta tirar foto e envia para a fila da IA
                    if estado_atual == "aguardando":
                        margem = CONFIG["MARGEM_ROSTO"]
                        recorte = frame[max(0, y1-margem):min(frame.shape[0], y2+margem), max(0, x1-margem):min(frame.shape[1], x2+margem)]
                        
                        if recorte.size > 0:
                            estado_rostos[track_id]["estado"] = "processando" 
                            fila_fotos_ia.put((track_id, recorte.copy())) 

                    # 2. IA retornou positivo: Libera a catraca virtual e grava visita
                    elif estado_atual == "reconhecido":
                        if not estado_rostos[track_id]["visita_registada"]:
                            db_id = estado_rostos[track_id]["db_id"]
                            if registrar_visita_hoje(conexao_db, db_id):
                                self.total_visitantes_hoje += 1
                                
                            estado_rostos[track_id]["visita_registada"] = True
                            mapa_dados_tela[track_id] = {"nome": estado_rostos[track_id]["nome"]}

                    # LÓGICA DE DESENHO (Cores baseadas no estado):
                    if track_id in mapa_dados_tela:
                        nome = mapa_dados_tela[track_id]["nome"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Verde (Liberado)
                        cv2.putText(frame, f"Liberado: {nome}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                    elif estado_atual == "processando":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2) # Laranja (Pensando)
                        cv2.putText(frame, "A analisar...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        
                    elif estado_atual == "na_fila":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Vermelho (Bloqueado)
                        cv2.putText(frame, "Aguardando Registo", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Informação do contador de visitas
            cv2.putText(frame, f"Visitantes Hoje: {self.total_visitantes_hoje}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Conversão de formato: OpenCV (BGR) -> Pillow -> Tkinter (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.lbl_video.imgtk = img_tk 
            self.lbl_video.configure(image=img_tk)

        # Loop: Chama esta mesma função infinitamente baseado no FPS configurado
        self.root.after(CONFIG["FPS_TELA"], self.atualizar_video)

    def atualizar_fila(self):
        """Verifica o dicionário de pendências e cria os botões na barra lateral."""
        for track_id, dados in list(cadastros_pendentes.items()):
            if dados["widget"] is None:
                # Cria o 'Cartão' (Frame) na interface
                card = tk.Frame(self.frame_dir, bg="#3e3e3e", pady=5, padx=5)
                card.pack(fill=tk.X, pady=5, padx=10)

                # Mostra a foto miniatura do rosto da pessoa
                img_cv = cv2.cvtColor(dados["foto"], cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_cv).resize((80, 80))
                img_tk = ImageTk.PhotoImage(image=img_pil)

                lbl_img = tk.Label(card, image=img_tk, bg="#3e3e3e")
                lbl_img.image = img_tk 
                lbl_img.pack(side=tk.LEFT)

                # Cria os textos e o botão de ação
                info_frame = tk.Frame(card, bg="#3e3e3e")
                info_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
                tk.Label(info_frame, text="Rosto Desconhecido", fg="white", bg="#3e3e3e", font=("Arial", 10)).pack(anchor="w")
                
                # O comando 'lambda' passa o ID específico da pessoa para a janela de registo
                btn = tk.Button(info_frame, text="Resolver Acesso", bg="orange", fg="black", font=("Arial", 9, "bold"),
                                command=lambda t=track_id: self.abrir_janela_cadastro(t))
                btn.pack(anchor="w", pady=5)

                dados["widget"] = card # Salva referência para podermos destruir o cartão mais tarde

        # Verifica a fila a cada 1 segundo (não precisa ser super rápido)
        self.root.after(1000, self.atualizar_fila) 

    def abrir_janela_cadastro(self, track_id):
        """Abre a janela Pop-up de registo com dupla função (Novo Registo ou Atualizar Foto)."""
        if track_id not in cadastros_pendentes: return
        
        janela = tk.Toplevel(self.root)
        janela.title("Resolução de Acesso")
        janela.geometry("350x350")
        janela.attributes('-topmost', True) # Força a ficar sempre à frente

        # Lógica de alternância de abas (Radio Buttons)
        modo_var = tk.StringVar(value="novo")
        def alternar_modo():
            if modo_var.get() == "novo":
                frame_busca.pack_forget()
                frame_novo.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            else:
                frame_novo.pack_forget()
                frame_busca.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # UI: Botões de seleção
        frame_opcoes = tk.Frame(janela)
        frame_opcoes.pack(pady=10)
        tk.Radiobutton(frame_opcoes, text="Novo Visitante", variable=modo_var, value="novo", command=alternar_modo).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(frame_opcoes, text="Já Possui Registo", variable=modo_var, value="busca", command=alternar_modo).pack(side=tk.LEFT, padx=10)

        # UI: Aba 1 - Novo Visitante
        frame_novo = tk.Frame(janela)
        tk.Label(frame_novo, text="Nome Completo:").pack(pady=(5,0))
        entry_nome = tk.Entry(frame_novo, width=30)
        entry_nome.pack(pady=5)
        tk.Label(frame_novo, text="RG / CPF:").pack()
        entry_doc = tk.Entry(frame_novo, width=30)
        entry_doc.pack(pady=5)

        # UI: Aba 2 - Busca por Documento (Falsa Rejeição)
        frame_busca = tk.Frame(janela)
        tk.Label(frame_busca, text="Digite o RG ou CPF:").pack(pady=(5,0))
        entry_busca_doc = tk.Entry(frame_busca, width=30)
        entry_busca_doc.pack(pady=5)
        lbl_erro_busca = tk.Label(frame_busca, text="", fg="red")
        lbl_erro_busca.pack()

        frame_novo.pack(fill=tk.BOTH, expand=True, padx=20, pady=10) # Abre por defeito a aba 1

        def processar_salvamento():
            """Executa a lógica de banco de dados dependendo da opção escolhida pelo utilizador."""
            assinatura = cadastros_pendentes[track_id]["assinatura"]
            
            if modo_var.get() == "novo":
                # --- FLUXO DE NOVO CADASTRO ---
                nome = entry_nome.get().strip() or "Desconhecido"
                doc = entry_doc.get().strip() or "N/A"
                
                pessoa_id = salvar_nova_pessoa(conexao_db, assinatura, nome, doc)
                memoria_pessoas.append((pessoa_id, nome, assinatura))
                
            else:
                # --- FLUXO DE APRENDIZADO DE MÁQUINA (Atualizar Rosto) ---
                doc_busca = entry_busca_doc.get().strip()
                resultado = buscar_pessoa_por_documento(conexao_db, doc_busca)
                
                if resultado:
                    pessoa_id, nome = resultado
                    
                    # Salva o novo rosto por cima do velho no banco de dados
                    atualizar_assinatura_pessoa(conexao_db, pessoa_id, assinatura)
                    
                    # Atualiza a memória RAM na mesma hora para o vídeo refletir a mudança
                    for i, (mem_id, mem_nome, mem_ass) in enumerate(memoria_pessoas):
                        if mem_id == pessoa_id:
                            memoria_pessoas[i] = (pessoa_id, nome, assinatura)
                            break
                else:
                    lbl_erro_busca.config(text="Documento não encontrado!")
                    return # Interrompe a função para o utilizador tentar de novo

            # Finalização padrão para qualquer um dos caminhos (Libera a catraca virtual)
            if registrar_visita_hoje(conexao_db, pessoa_id):
                self.total_visitantes_hoje += 1

            estado_rostos[track_id]["estado"] = "reconhecido"
            estado_rostos[track_id]["db_id"] = pessoa_id
            estado_rostos[track_id]["nome"] = nome
            estado_rostos[track_id]["visita_registada"] = True
            mapa_dados_tela[track_id] = {"nome": nome}

            # Remove da fila de espera e fecha a aba Pop-up
            cadastros_pendentes[track_id]["widget"].destroy()
            del cadastros_pendentes[track_id]
            janela.destroy()

        tk.Button(janela, text="Confirmar Acesso", command=processar_salvamento, bg="green", fg="white", font=("Arial", 10, "bold")).pack(pady=10)


# ======================================================================
# 🚀 PONTO DE ENTRADA DO PROGRAMA
# ======================================================================
if __name__ == "__main__":
    print("A inicializar os módulos do Sistema de Acesso...")
    conexao_db = iniciar_banco()
    memoria_pessoas = carregar_todas_pessoas(conexao_db)

    root = tk.Tk()
    app = CentroDeComandoApp(root)
    root.state('zoomed') # Inicia maximizado
    root.mainloop()      # Trava o Python aqui rodando a interface gráfica

    # Quando o utilizador fechar a janela no 'X', o código abaixo é executado para limpar a memória
    print("A encerrar sistema de forma segura...")
    fila_fotos_ia.put(None) # Manda a IA parar
    app.cap.release()       # Desliga a luz da câmara
    conexao_db.close()      # Salva e fecha o banco de dados