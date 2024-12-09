# Stable Diffusion Inpainting para Brasileirização de Imagens

Este notebook utiliza o modelo Stable Diffusion Inpainting para modificar personagens humanos em imagens para que se pareçam mais com brasileiros urbanos, mantendo o estilo e a composição da cena originais. O notebook é projetado para processar imagens em lote localizadas em um diretório especificado do Google Drive.

## Pré-requisitos

- Acesso ao Google Drive com permissões de leitura e gravação.
- Uma conta do Google com autenticação configurada para uso com o Google Colab e o Google Drive.
- Familiaridade com o Google Colab e notebooks Jupyter.
- Compreensão básica dos conceitos de difusão estável e inpainting.

## Configuração

### 1. Montagem do Google Drive

O notebook começa forçando a remontagem do Google Drive para garantir que quaisquer alterações nos arquivos sejam refletidas e que o cache local seja limpo.

```python
if os.path.exists('/content/drive'):
    drive.flush_and_unmount()
    print('Google Drive desmontado.')

drive.mount('/content/drive', force_remount=True)
print('Google Drive montado com sucesso!')
```

### 2. Autenticação

Autentica o usuário e inicializa a biblioteca `gspread` para interação com o Google Sheets, se necessário.

```python
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
print('Autenticação no Google Drive realizada com sucesso!')
```

### 3. Configuração de Diretórios

Define os diretórios de entrada e saída no Google Drive e cria o diretório de saída, se ele não existir.

```python
INPUT_DIR = os.path.join('/content/drive/My Drive/input-images')
OUTPUT_DIR = os.path.join('/content/drive/My Drive/output-images')

os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### 4. Verificação do Diretório de Entrada

Verifica se o diretório de entrada existe. Se não existir, ele cria o diretório e instrui o usuário a adicionar imagens a ele.

```python
if not os.path.exists(INPUT_DIR):
    print(f"Erro: Diretório de entrada '{INPUT_DIR}' não encontrado.")
    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f"Diretório de entrada '{INPUT_DIR}' criado. Adicione as imagens e execute novamente.")
```

## Processamento de Imagens

### 1. Configuração do Pipeline

Configura o pipeline `StableDiffusionInpaintPipeline` com o modelo pré-treinado `runwayml/stable-diffusion-inpainting`.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to(device)
```

### 2. Definição do Prompt

Define o prompt que guiará o processo de inpainting. O prompt instrui o modelo a gerar imagens hiper-realistas e modificar os personagens humanos para que se pareçam mais com brasileiros urbanos, mantendo o estilo e a composição originais.

```python
PROMPT = (
    "As imagens originais são hiper-realistas. As imagens que você gerar devem ser igualmente hiper-realistas. mantendo o estilo e a composição da cena intactos, faça com que os personagens humanos na imagem se pareçam mais com brasileiros urbanos."
)
```

### 3. Função de Processamento de Imagens

Define uma função `process_images` para processar cada imagem no diretório de entrada.

```python
def process_images(input_dir, output_dir, prompt):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_output.png")

            try:
                print(f"Processando: {filename}")
                init_image = Image.open(input_path).convert("RGB").resize((1280, 720))
                mask_image = Image.new("RGB", init_image.size, "white")

                result = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50).images[0]

                result.save(output_path)
                print(f"Imagem salva em: {output_path}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
```

### 4. Execução do Processamento

Chama a função `process_images` para processar todas as imagens no diretório de entrada.

```python
process_images(INPUT_DIR, OUTPUT_DIR, PROMPT)
```

## Uso

1. Certifique-se de que seu Google Drive esteja montado e que você tenha configurado os diretórios de entrada e saída.
2. Coloque as imagens que deseja processar no diretório `INPUT_DIR`.
3. Execute o notebook sequencialmente para processar as imagens.
4. As imagens processadas serão salvas no diretório `OUTPUT_DIR` com o sufixo `_output.png`.

## Solução de Problemas

- Se o diretório de entrada não for encontrado, o notebook o criará. Adicione suas imagens ao diretório criado e execute o notebook novamente.
- Quaisquer erros durante o processamento de imagens serão impressos no console. Verifique a mensagem de erro para obter detalhes.

## Observações

- O notebook usa o dispositivo `cuda` se disponível; caso contrário, ele usa a `cpu`.
- O pipeline de inpainting usa imagens redimensionadas para 1280x720 pixels.
- A imagem de máscara é uma imagem branca, indicando que toda a imagem pode ser modificada pelo processo de inpainting.
- O número de etapas de inferência é definido como 50. Você pode ajustar esse valor, se necessário.

Este notebook fornece uma solução simples para modificar personagens humanos em imagens para que se pareçam mais com brasileiros urbanos usando o modelo Stable Diffusion Inpainting.
