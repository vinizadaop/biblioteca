"""
Carrega o modelo Zephyr 7B Beta da Hugging Face e gera respostas a partir de prompts do usuário.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Nome do modelo no Hugging Face Hub
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

def carregar_modelo(model_name):
    """
    Carrega o modelo e o tokenizer.
    """
    print("Carregando o modelo... Isso pode levar alguns minutos.")

    # Detecta se há uma GPU disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega o tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Carrega o modelo com mapeamento para o dispositivo apropriado
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model.to(device)
    print(f"Modelo carregado com sucesso em: {device}")
    return model, tokenizer

def gerar_resposta(prompt, model, tokenizer):
    """
    Gera uma resposta a partir do prompt usando o modelo carregado.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """
    Executa o loop principal do programa para interação com o usuário.
    """
    try:
        model, tokenizer = carregar_modelo(MODEL_NAME)

        while True:
            prompt = input("\nDigite seu prompt (ou 'sair' para encerrar): ")
            if prompt.strip().lower() == "sair":
                print("Encerrando o programa.")
                break

            resposta = gerar_resposta(prompt, model, tokenizer)
            print("\nResposta gerada:\n", resposta)

    except (OSError, ValueError, RuntimeError) as e:  # Replace with specific exceptions
        print(f"\nOcorreu um erro: {e}")

if __name__ == "__main__":
    main()
