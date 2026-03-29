import sqlite3
from datetime import date, timedelta

print("=== MÁQUINA DO TEMPO: INJETOR DE HISTÓRICO ===")

# 1. Conecta ao nosso banco de dados
conn = sqlite3.connect('controle_acesso.db')
cursor = conn.cursor()

# 2. Descobre qual é o seu ID (vamos pegar o ID 1, que deve ser o seu)
pessoa_id = 1

# Verifica se o ID 1 existe antes de injetar
cursor.execute("SELECT nome FROM pessoas WHERE id = ?", (pessoa_id,))
resultado = cursor.fetchone()

if resultado is None:
    print(f"Erro: A pessoa com ID {pessoa_id} ainda não está cadastrada.")
    print("Por favor, rode o contador.py primeiro e faça o seu cadastro na câmera.")
else:
    nome = resultado[0]
    print(f"Injetando histórico falso para: {nome}")
    print("-" * 30)

    # 3. Faz um loop voltando no tempo (de 1 a 6 dias atrás)
    hoje = date.today()
    
    for dias_atras in range(1, 7):
        data_passada = hoje - timedelta(days=dias_atras)
        data_str = data_passada.isoformat()
        hora_falsa = "08:30:00" # Uma hora qualquer de entrada
        
        # Insere a visita diretamente na tabela
        cursor.execute("INSERT INTO visitas (pessoa_id, data_visita, hora_visita) VALUES (?, ?, ?)",
                      (pessoa_id, data_str, hora_falsa))
        
        print(f"-> Visita simulada em: {data_str}")

    conn.commit()
    print("-" * 30)
    print("Sucesso! Histórico gerado. Pode abrir a sua câmera agora.")

conn.close()