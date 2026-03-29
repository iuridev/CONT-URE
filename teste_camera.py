import cv2

# Tenta abrir a câmera 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERRO: O sistema não encontrou a câmera no índice 0.")
else:
    print("Câmera encontrada! Pressione 'q' na janela do vídeo para fechar.")

while cap.isOpened():
    sucesso, frame = cap.read()
    
    if not sucesso:
        print("ERRO: A câmera está aberta, mas não está enviando imagens (pode estar em uso por outro app).")
        break
        
    # Mostra a imagem pura da câmera
    cv2.imshow("Teste de Camera Puro", frame)

    # Aperte 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()