import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from sqlalchemy import create_engine, Column, Integer, String, Date, Time, ForeignKey, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Configuração da Página Web
st.set_page_config(page_title="Dashboard de Acessos", page_icon="📊", layout="wide")

# ==========================================
# 1. ARQUITETURA ORM (SQLAlchemy)
# ==========================================
# Aqui nós "ensinamos" ao Python como as nossas tabelas são por dentro
Base = declarative_base()

class Pessoa(Base):
    __tablename__ = 'pessoas'
    id = Column(Integer, primary_key=True, autoincrement=True)
    nome = Column(String)
    documento = Column(String)
    assinatura_facial = Column(String)
    
    # Cria a ligação (relacionamento) com o histórico de visitas
    visitas = relationship("Visita", back_populates="pessoa")

class Visita(Base):
    __tablename__ = 'visitas'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pessoa_id = Column(Integer, ForeignKey('pessoas.id'))
    data_visita = Column(String) # No SQLite salvamos como texto (YYYY-MM-DD)
    hora_visita = Column(String)
    
    pessoa = relationship("Pessoa", back_populates="visitas")

# Conectamos ao banco existente
engine = create_engine('sqlite:///controle_acesso.db')
Session = sessionmaker(bind=engine)

# ==========================================
# 2. BUSCA DE DADOS (Sem Raw SQL)
# ==========================================
def get_dados():
    session = Session()
    hoje = date.today().isoformat()
    
    # Busca 1: Visitas agrupadas por Mês
    query_meses = session.query(
        func.strftime('%m/%Y', Visita.data_visita).label('mes'),
        func.count(Visita.id).label('total_visitas')
    ).group_by(
        func.strftime('%Y-%m', Visita.data_visita)
    ).order_by(
        func.strftime('%Y-%m', Visita.data_visita).desc()
    ).limit(12)
    
    # Busca 2: Top Visitantes (Unindo a tabela Pessoa e Visita)
    query_top = session.query(
        Pessoa.nome,
        func.count(Visita.id).label('total_visitas')
    ).join(Visita).group_by(Pessoa.id).order_by(func.count(Visita.id).desc()).limit(10)
    
    # Busca 3: Visitantes de Hoje
    visitantes_hoje = session.query(func.count(func.distinct(Visita.pessoa_id)))\
                             .filter(Visita.data_visita == hoje).scalar()
    
    # Converte as buscas seguras do ORM para tabelas do Pandas (DataFrames)
    df_meses = pd.read_sql(query_meses.statement, session.bind)
    df_top = pd.read_sql(query_top.statement, session.bind)
    
    session.close()
    return df_meses, df_top, visitantes_hoje

# Carrega os dados
df_meses, df_top, visitantes_hoje = get_dados()

# ==========================================
# 3. CONSTRUÇÃO DO VISUAL DO SITE
# ==========================================
st.title("📊 Painel de Controle de Acessos")
st.markdown("---")

# Métricas Rápidas
col1, col2, col3 = st.columns(3)
col1.metric("Visitantes Únicos Hoje", visitantes_hoje)
col2.metric("Total de Entradas (Histórico)", df_meses['total_visitas'].sum() if not df_meses.empty else 0)
col3.metric("Pessoas Registadas no Sistema", len(df_top) if not df_top.empty else 0)

st.markdown("---")

col_grafico1, col_grafico2 = st.columns(2)

with col_grafico1:
    st.subheader("📈 Visitas nos Últimos 12 Meses")
    if not df_meses.empty:
        df_meses = df_meses.iloc[::-1] # Inverte para ordem cronológica
        fig1 = px.bar(df_meses, x='mes', y='total_visitas', 
                     labels={'mes': 'Mês', 'total_visitas': 'Acessos'},
                     color_discrete_sequence=['#00a6ff'])
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Aguardando dados para gerar o gráfico.")

with col_grafico2:
    st.subheader("🏆 Top Visitantes (Mais Assíduos)")
    if not df_top.empty:
        fig2 = px.pie(df_top, names='nome', values='total_visitas', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Aguardando dados para gerar o gráfico.")

# Tabela Detalhada
st.subheader("📋 Lista de Visitantes Frequentes")
if not df_top.empty:
    st.dataframe(df_top, use_container_width=True)