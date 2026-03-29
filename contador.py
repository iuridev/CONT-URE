"""
SISTEMA DE ACESSO LIVRE (FREE FLOW) COM FILA DE ATENDIMENTO E AUTO-SETUP
Arquitetura: 
- Visão Computacional: OpenCV (Captura) + YOLO (Rastreio de Objetos)
- Biometria: DeepFace (Extração de características faciais)
- GUI: Tkinter (Interface de utilizador nativa)
- Concorrência: Threads + Queue (Processamento assíncrono para não travar o vídeo)
- Persistência: SQLAlchemy (ORM corporativo compatível com MySQL, PostgreSQL e SQLite)
"""

import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import json
from datetime import date, datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading 
import queue     

# Bibliotecas do SQLAlchemy (Mapeamento Objeto-Relacional). 
# Permitem interagir com o banco de dados usando objetos Python em vez de comandos SQL crus.
from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.orm import declarative_base, sessionmaker

# Desativa avisos de compilação do TensorFlow (motor por trás do DeepFace) para manter o terminal limpo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ARQUIVO_CONFIG = 'config.json'

# ======================================================================
# ⚙️ ÁREA DE CONFIGURAÇÕES AVANÇADAS
# ======================================================================
# Variáveis globais que ajustam o comportamento "fino" do sistema.
CONFIG = {
    "MODELO_YOLO": 'yolov8n.pt',      # Modelo 'nano' do YOLOv8 (Foco em velocidade de inferência)
    "TOLERANCIA_FACIAL": 0.55,        # Limiar de distância cosseno. Valores menores exigem maior semelhança (mais rigor).
    "FPS_TELA": 15,                   # Intervalo em ms para atualizar o Tkinter (15ms ~= 60 Frames por Segundo).
    "MARGEM_ROSTO": 15                # Margem de segurança (pixels) ao recortar o rosto para a IA ler melhor.
}

# ======================================================================
# 🛠️ ASSISTENTE DE PRIMEIRA EXECUÇÃO (SETUP WIZARD)
# ======================================================================
def executar_setup_inicial():
    """
    Interface gráfica disparada apenas na primeira vez que o programa roda (ou se o config.json sumir).
    Gera as credenciais de acesso ao banco de dados e escolhe a câmera.
    """
    setup_janela = tk.Tk()
    setup_janela.title("Setup Inicial - Servidor e Câmera")
    setup_janela.geometry("500x450")
    setup_janela.configure(bg="#f0f0f0")
    setup_janela.eval('tk::PlaceWindow . center') # Centraliza a janela no ecrã

    tk.Label(setup_janela, text="Configuração do Sistema", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

    # Variável Tkinter que armazena a escolha do utilizador (MySQL ou SQLite)
    tk.Label(setup_janela, text="1. Escolha o tipo de Banco de Dados:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor="w", padx=20)
    tipo_bd = tk.StringVar(value="mysql") 
    
    frame_radios = tk.Frame(setup_janela, bg="#f0f0f0")
    frame_radios.pack(fill="x", padx=20, pady=5)
    tk.Radiobutton(frame_radios, text="Servidor MySQL da Empresa (Recomendado)", variable=tipo_bd, value="mysql", bg="#f0f0f0").pack(anchor="w")
    tk.Radiobutton(frame_radios, text="Local (Testes sem Internet - SQLite)", variable=tipo_bd, value="sqlite_local", bg="#f0f0f0").pack(anchor="w")

    tk.Label(setup_janela, text="Se escolheu MySQL, preencha abaixo:", bg="#f0f0f0", fg="blue").pack(anchor="w", padx=20, pady=(10,0))
    tk.Label(setup_janela, text="Formato: usuario:senha@IP/banco", bg="#f0f0f0").pack(anchor="w", padx=20)
    
    entry_mysql = tk.Entry(setup_janela, width=50)
    entry_mysql.insert(0, "usuario:senha@10.180.119.13/controle_acesso")
    entry_mysql.pack(padx=20, pady=5)

    tk.Label(setup_janela, text="2. ID da Câmera (0 = Padrão, 1 = USB Externa):", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor="w", padx=20, pady=(15,0))
    entry_camera = tk.Entry(setup_janela, width=10)
    entry_camera.insert(0, "0")
    entry_camera.pack(anchor="w", padx=20, pady=5)

    def salvar_configuracoes():
        """Lê os campos da tela e escreve o ficheiro config.json"""
        if tipo_bd.get() == "sqlite_local":
            string_conexao = 'sqlite:///controle_acesso.db'
        else:
            # Padrão de string de conexão do SQLAlchemy para MySQL
            credenciais = entry_mysql.get().strip()
            string_conexao = f'mysql+pymysql://{credenciais}'

        config = {
            "DATABASE_URI": string_conexao,
            "CAMERA_ID": int(entry_camera.get())
        }

        with open(ARQUIVO_CONFIG, 'w') as f:
            json.dump(config, f, indent=4) # Grava formatado
        
        messagebox.showinfo("Sucesso", "Sistema configurado! O aplicativo será iniciado.")
        setup_janela.destroy() # Fecha a tela de setup e permite que o programa continue

    tk.Button(setup_janela, text="Salvar e Ligar Sistema", command=salvar_configuracoes, bg="green", fg="white", font=("Arial", 10, "bold")).pack(pady=20)
    setup_janela.mainloop()

# ======================================================================
# 🛡️ LÓGICA DE BOOT (Auto-Cura)
# ======================================================================
def carregar_configuracoes():
    """Garante que o programa não trava se o config.json estiver vazio ou corrompido."""
    if not os.path.exists(ARQUIVO_CONFIG):
        return False
    try:
        with open(ARQUIVO_CONFIG, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return False 

# ======================================================================
# 💾 MÓDULO DE BANCO DE DADOS CORPORATIVO (SQLAlchemy)
# ======================================================================
Base = declarative_base() # Classe base obrigatória para criar os modelos ORM

# --- MODELOS DE DADOS (Tabelas do Banco) ---
class Pessoa(Base):
    """Representa a tabela 'pessoas' no banco de dados."""
    __tablename__ = 'pessoas'
    id = Column(Integer, primary_key=True, autoincrement=True)
    nome = Column(String(100))
    documento = Column(String(50))
    # O vetor biométrico do DeepFace tem várias dimensões. Guardamos como uma string JSON longa.
    assinatura_facial = Column(String(5000)) 

class Visita(Base):
    """Representa a tabela 'visitas' no banco de dados."""
    __tablename__ = 'visitas'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pessoa_id = Column(Integer)
    data_visita = Column(String(20))
    hora_visita = Column(String(20))

# --- FUNÇÕES DE TRANSAÇÃO (CRUD) ---
def iniciar_banco(uri):
    """Cria a conexão ('engine') e gera as tabelas automaticamente se não existirem."""
    engine = create_engine(uri)
    Base.metadata.create_all(engine) 
    Session = sessionmaker(bind=engine)
    return Session() # Retorna a porta de comunicação com o banco

def carregar_todas_pessoas(session):
    """Faz cache (carrega para a RAM) de todas as biometrias para a IA poder consultar rápido."""
    pessoas = session.query(Pessoa).all()
    # Converte o JSON guardado no banco de volta para listas matemáticas
    return [(p.id, p.nome, json.loads(p.assinatura_facial)) for p in pessoas]

def salvar_nova_pessoa(session, vetor, nome, doc):
    nova_pessoa = Pessoa(nome=nome, documento=doc, assinatura_facial=json.dumps(vetor))
    session.add(nova_pessoa) # Prepara o envio
    session.commit()         # Executa e salva de facto
    return nova_pessoa.id

def buscar_pessoa_por_documento(session, documento):
    # .first() retorna o primeiro resultado que encontrar (ou None)
    pessoa = session.query(Pessoa).filter_by(documento=documento).first()
    if pessoa:
        return (pessoa.id, pessoa.nome)
    return None

def atualizar_assinatura_pessoa(session, pessoa_id, nova_assinatura):
    """Atualiza a foto de alguém existente. Lógica de 'Machine Learning Contínuo'."""
    pessoa = session.query(Pessoa).filter_by(id=pessoa_id).first()
    if pessoa:
        pessoa.assinatura_facial = json.dumps(nova_assinatura)
        session.commit()

def registrar_visita_hoje(session, pessoa_id):
    hoje = date.today().isoformat()
    agora = datetime.now().strftime("%H:%M:%S")
    # Verifica se a pessoa JÁ ENTROU hoje para não duplicar acessos
    visita_existente = session.query(Visita).filter_by(pessoa_id=pessoa_id, data_visita=hoje).first()
    
    if not visita_existente: 
        nova_visita = Visita(pessoa_id=pessoa_id, data_visita=hoje, hora_visita=agora)
        session.add(nova_visita)
        session.commit()
        return True 
    return False

def contar_visitantes_hoje(session):
    hoje = date.today().isoformat()
    # Query SQL traduzida: SELECT COUNT(DISTINCT pessoa_id) FROM visitas WHERE data_visita = hoje
    return session.query(func.count(func.distinct(Visita.pessoa_id))).filter(Visita.data_visita == hoje).scalar()

# ======================================================================
# 🧠 MÓDULO DE INTELIGÊNCIA ARTIFICIAL (Roda em 2º Plano)
# ======================================================================
# Fila Thread-Safe: Única forma segura da câmera (Thread principal) enviar fotos para a IA (Thread secundária)
fila_fotos_ia = queue.Queue()

# Dicionários de Memória Partilhada
estado_rostos = {}        # Guarda o ID do YOLO e o status (Aguardando, Processando, Reconhecido, Na Fila)
cadastros_pendentes = {}  # Guarda a foto recortada de desconhecidos para mostrar na barra lateral
mapa_dados_tela = {}      # Usado para desenhar o texto verde em cima da cabeça da pessoa

def trabalhador_ia():
    """Worker eterno: Fica aguardando fotos chegarem na fila para fazer a matemática biométrica."""
    while True:
        tarefa = fila_fotos_ia.get() # O código pausa aqui até chegar uma foto
        if tarefa is None: break     # Comando especial (None) para desligar a Thread no encerramento
        
        track_id, recorte_rosto = tarefa
        
        try:
            # Extrai o vetor de 128 dimensões (Facenet) do rosto recortado
            embedded = DeepFace.represent(img_path=recorte_rosto, model_name='Facenet', enforce_detection=True, detector_backend='mtcnn', align=True)
            nova_assinatura = embedded[0]["embedding"]
            
            reconhecido = False
            # Compara a nova foto com todas as fotos salvas na memória RAM
            for db_id, nome_banco, assinatura_salva in memoria_pessoas:
                a, b = np.array(nova_assinatura), np.array(assinatura_salva)
                # Cálculo de Similaridade por Cosseno (Compara a direção dos vetores faciais)
                dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

                if dist < CONFIG["TOLERANCIA_FACIAL"]:
                    # Match encontrado!
                    estado_rostos[track_id]["estado"] = "reconhecido"
                    estado_rostos[track_id]["db_id"] = db_id
                    estado_rostos[track_id]["nome"] = nome_banco
                    estado_rostos[track_id]["assinatura"] = nova_assinatura
                    reconhecido = True
                    break
            
            if not reconhecido:
                # Pessoa não existe. Coloca na fila de interface (barra lateral)
                estado_rostos[track_id]["estado"] = "na_fila"
                cadastros_pendentes[track_id] = {
                    "assinatura": nova_assinatura,
                    "foto": recorte_rosto.copy(),
                    "widget": None # Placeholder para o elemento de UI que será criado depois
                }
                
        except Exception as e:
            print(f"🚨 ERRO NA IA DE BIOMETRIA: {e}") # <-- ADICIONÁMOS ESTE PRINT
            estado_rostos[track_id]["estado"] = "aguardando"
            
        fila_fotos_ia.task_done() # Avisa à fila que o trabalho terminou


# ======================================================================
# 🖥️ MÓDULO DE INTERFACE GRÁFICA E CÂMARA (Centro de Comando)
# ======================================================================
class CentroDeComandoApp:
    def __init__(self, root, config_sistema):
        """Construtor da Janela Principal."""
        self.root = root
        self.root.title("Centro de Comando - Acesso Livre (SQL Server)")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.total_visitantes_hoje = contar_visitantes_hoje(sessao_db)

        # Divisão da Tela: Frame Esquerdo (Câmera) e Direito (Cadastros pendentes)
        self.frame_esq = tk.Frame(self.root, bg="#000000", width=800, height=600)
        self.frame_esq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_dir = tk.Frame(self.root, bg="#2d2d2d", width=350)
        self.frame_dir.pack(side=tk.RIGHT, fill=tk.Y)
        self.frame_dir.pack_propagate(False) # Força a largura do frame a não encolher

        tk.Label(self.frame_dir, text="Fila de Registo", font=("Arial", 16, "bold"), bg="#2d2d2d", fg="white").pack(pady=10)

        # Rótulo de imagem que funcionará como a nossa "Tela de Cinema" para o OpenCV
        self.lbl_video = tk.Label(self.frame_esq, bg="black")
        self.lbl_video.pack(expand=True, fill=tk.BOTH)

        # Inicializa Modelos AI
        self.model = YOLO(CONFIG["MODELO_YOLO"])
        self.cap = cv2.VideoCapture(config_sistema["CAMERA_ID"])
        
        # Desperta a Thread da IA criada no módulo anterior
        self.thread_ia = threading.Thread(target=trabalhador_ia, daemon=True)
        self.thread_ia.start()

        # Inicia os "Relógios" (Loops) da Interface Gráfica
        self.atualizar_video()
        self.atualizar_fila()

    def atualizar_video(self):
        """Lê frames da webcam, passa no YOLO e desenha na tela."""
        sucesso, frame = self.cap.read()
        if sucesso:
            # track() mantém o ID consistente enquanto a pessoa se move no vídeo
            resultados = self.model.track(frame, persist=True, classes=[0], verbose=False)

            if resultados[0].boxes.id is not None:
                boxes = resultados[0].boxes.xyxy.cpu()
                track_ids = resultados[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Se um ID novo apareceu na tela, regista-o no estado inicial
                    if track_id not in estado_rostos:
                        estado_rostos[track_id] = {"estado": "aguardando", "db_id": None, "nome": "", "assinatura": None, "visita_registada": False}
                    
                    estado_atual = estado_rostos[track_id]["estado"]
                    
                    # --- MÁQUINA DE ESTADOS DA PESSOA ---
                    if estado_atual == "aguardando":
                        margem = CONFIG["MARGEM_ROSTO"]
                        recorte = frame[max(0, y1-margem):min(frame.shape[0], y2+margem), max(0, x1-margem):min(frame.shape[1], x2+margem)]
                        
                        if recorte.size > 0:
                            estado_rostos[track_id]["estado"] = "processando" 
                            fila_fotos_ia.put((track_id, recorte.copy())) # Joga para a Thread de Fundo

                    elif estado_atual == "reconhecido":
                        # Pessoa acabou de ser reconhecida: Libera a visita no banco
                        if not estado_rostos[track_id]["visita_registada"]:
                            db_id = estado_rostos[track_id]["db_id"]
                            if registrar_visita_hoje(sessao_db, db_id):
                                self.total_visitantes_hoje += 1 # Sobe o contador do ecrã
                                
                            estado_rostos[track_id]["visita_registada"] = True
                            mapa_dados_tela[track_id] = {"nome": estado_rostos[track_id]["nome"]}

                    # --- LÓGICA DE DESENHO (Visual Feedback) ---
                    if track_id in mapa_dados_tela:
                        nome = mapa_dados_tela[track_id]["nome"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Verde
                        cv2.putText(frame, f"Liberado: {nome}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                    elif estado_atual == "processando":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2) # Laranja
                        cv2.putText(frame, "A analisar...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        
                    elif estado_atual == "na_fila":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Vermelho
                        cv2.putText(frame, "Aguardando Registo", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Desenha contador global
            cv2.putText(frame, f"Visitantes Hoje: {self.total_visitantes_hoje}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Magia de UI: Converte formato de imagem do OpenCV (BGR) para o formato exigido pelo Tkinter (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.lbl_video.imgtk = img_tk 
            self.lbl_video.configure(image=img_tk)

        # O Tkinter não usa while True. Usamos .after() para ele chamar a função repetidamente.
        self.root.after(CONFIG["FPS_TELA"], self.atualizar_video)

    def atualizar_fila(self):
        """Lê o dicionário de pendentes e desenha pequenos "cartões" de perfil na barra direita."""
        for track_id, dados in list(cadastros_pendentes.items()):
            if dados["widget"] is None:
                # Frame base do cartão
                card = tk.Frame(self.frame_dir, bg="#3e3e3e", pady=5, padx=5)
                card.pack(fill=tk.X, pady=5, padx=10)

                # Processa a foto miniatura do rosto
                img_cv = cv2.cvtColor(dados["foto"], cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_cv).resize((80, 80))
                img_tk = ImageTk.PhotoImage(image=img_pil)

                lbl_img = tk.Label(card, image=img_tk, bg="#3e3e3e")
                lbl_img.image = img_tk 
                lbl_img.pack(side=tk.LEFT)

                info_frame = tk.Frame(card, bg="#3e3e3e")
                info_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
                tk.Label(info_frame, text="Rosto Desconhecido", fg="white", bg="#3e3e3e", font=("Arial", 10)).pack(anchor="w")
                
                # lambda força o botão a lembrar qual track_id (ID da pessoa) ele representa
                btn = tk.Button(info_frame, text="Resolver Acesso", bg="orange", fg="black", font=("Arial", 9, "bold"),
                                command=lambda t=track_id: self.abrir_janela_cadastro(t))
                btn.pack(anchor="w", pady=5)

                dados["widget"] = card # Salva a referência na memória para conseguirmos deletar o widget no futuro

        self.root.after(1000, self.atualizar_fila) # Checa a fila a cada 1 segundo (Não precisa ser rápido como o vídeo)

    def abrir_janela_cadastro(self, track_id):
        """Abre Pop-Up para o rececionista cadastrar a pessoa. Não interrompe o vídeo principal!"""
        if track_id not in cadastros_pendentes: return
        
        janela = tk.Toplevel(self.root)
        janela.title("Resolução de Acesso")
        janela.geometry("350x350")
        janela.attributes('-topmost', True) 

        # --- Lógica de Abas (Novo vs Atualizar) ---
        modo_var = tk.StringVar(value="novo")
        def alternar_modo():
            """Esconde um frame e mostra o outro dependendo do RadioButton selecionado."""
            if modo_var.get() == "novo":
                frame_busca.pack_forget()
                frame_novo.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            else:
                frame_novo.pack_forget()
                frame_busca.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # UI: Topo
        frame_opcoes = tk.Frame(janela)
        frame_opcoes.pack(pady=10)
        tk.Radiobutton(frame_opcoes, text="Novo Visitante", variable=modo_var, value="novo", command=alternar_modo).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(frame_opcoes, text="Já Possui Registo", variable=modo_var, value="busca", command=alternar_modo).pack(side=tk.LEFT, padx=10)

        # UI: Aba 1 (Criar do Zero)
        frame_novo = tk.Frame(janela)
        tk.Label(frame_novo, text="Nome Completo:").pack(pady=(5,0))
        entry_nome = tk.Entry(frame_novo, width=30)
        entry_nome.pack(pady=5)
        tk.Label(frame_novo, text="RG / CPF:").pack()
        entry_doc = tk.Entry(frame_novo, width=30)
        entry_doc.pack(pady=5)

        # UI: Aba 2 (Buscar e Atualizar)
        frame_busca = tk.Frame(janela)
        tk.Label(frame_busca, text="Digite o RG ou CPF:").pack(pady=(5,0))
        entry_busca_doc = tk.Entry(frame_busca, width=30)
        entry_busca_doc.pack(pady=5)
        lbl_erro_busca = tk.Label(frame_busca, text="", fg="red")
        lbl_erro_busca.pack()

        frame_novo.pack(fill=tk.BOTH, expand=True, padx=20, pady=10) 

        def processar_salvamento():
            """Função engatilhada ao clicar em 'Confirmar Acesso'."""
            assinatura = cadastros_pendentes[track_id]["assinatura"]
            
            if modo_var.get() == "novo":
                nome = entry_nome.get().strip() or "Desconhecido"
                doc = entry_doc.get().strip() or "N/A"
                
                # Persistência DB e Atualização de RAM
                pessoa_id = salvar_nova_pessoa(sessao_db, assinatura, nome, doc)
                memoria_pessoas.append((pessoa_id, nome, assinatura))
                
            else:
                doc_busca = entry_busca_doc.get().strip()
                resultado = buscar_pessoa_por_documento(sessao_db, doc_busca)
                
                if resultado:
                    pessoa_id, nome = resultado
                    # Treino Contínuo: Sobrescreve biometria velha com a nova foto
                    atualizar_assinatura_pessoa(sessao_db, pessoa_id, assinatura)
                    
                    # Atualiza o array em RAM para reflexão imediata no sistema
                    for i, (mem_id, mem_nome, mem_ass) in enumerate(memoria_pessoas):
                        if mem_id == pessoa_id:
                            memoria_pessoas[i] = (pessoa_id, nome, assinatura)
                            break
                else:
                    lbl_erro_busca.config(text="Documento não encontrado!")
                    return # Interrompe fluxo para utilizador tentar de novo

            # Registra visita e soma contador
            if registrar_visita_hoje(sessao_db, pessoa_id):
                self.total_visitantes_hoje += 1

            # Muda estado do objeto YOLO na tela principal para "Verde/Reconhecido"
            estado_rostos[track_id]["estado"] = "reconhecido"
            estado_rostos[track_id]["db_id"] = pessoa_id
            estado_rostos[track_id]["nome"] = nome
            estado_rostos[track_id]["visita_registada"] = True
            mapa_dados_tela[track_id] = {"nome": nome}

            # Destrói o cartão da barra lateral e fecha o Pop-Up
            cadastros_pendentes[track_id]["widget"].destroy()
            del cadastros_pendentes[track_id]
            janela.destroy()

        tk.Button(janela, text="Confirmar Acesso", command=processar_salvamento, bg="green", fg="white", font=("Arial", 10, "bold")).pack(pady=10)


# ======================================================================
# 🚀 PONTO DE ENTRADA DO PROGRAMA (Main)
# ======================================================================
if __name__ == "__main__":
    print("A verificar configurações do sistema...")
    
    # 1. Carrega dados de configuração (Banco e Câmera)
    CONFIG_SISTEMA = carregar_configuracoes()

    if not CONFIG_SISTEMA:
        print("A abrir Assistente de Setup...")
        executar_setup_inicial()
        CONFIG_SISTEMA = carregar_configuracoes()
        
        # Aborta caso o utilizador feche o wizard sem configurar
        if not CONFIG_SISTEMA:
            print("Setup cancelado.")
            exit()

    print(f"Ligar câmara ID: {CONFIG_SISTEMA['CAMERA_ID']}")
    print(f"Ligar ao Servidor: {CONFIG_SISTEMA['DATABASE_URI']}")
    
    # 2. Conecta ao Banco (MySQL/SQLite) usando Bloco Try-Except para segurança
    try:
        sessao_db = iniciar_banco(CONFIG_SISTEMA['DATABASE_URI'])
        memoria_pessoas = carregar_todas_pessoas(sessao_db)
    except Exception as e:
        print("ERRO CRÍTICO: Não foi possível conectar ao Banco de Dados.")
        print("Verifique se a VPN está ligada, se o IP está correto e se o banco 'controle_acesso' foi criado no phpMyAdmin.")
        print(f"Detalhe técnico: {e}")
        os.remove(ARQUIVO_CONFIG) # Reseta o config para forçar setup no próximo boot
        exit()

    # 3. Dá boot na Interface de Janela Principal do Tkinter
    root = tk.Tk()
    app = CentroDeComandoApp(root, CONFIG_SISTEMA)
    root.state('zoomed') # Inicia Maximizado no Windows
    root.mainloop()      # Trava a thread principal aqui executando a GUI

    # 4. Sequência de Encerramento (Graceful Shutdown) disparada ao fechar a janela
    print("A encerrar sistema de forma segura...")
    fila_fotos_ia.put(None) # Manda 'Veneno' para matar a Thread da IA
    app.cap.release()       # Liberta o hardware da câmera
    sessao_db.close()       # Fecha conexão com o banco MySQL