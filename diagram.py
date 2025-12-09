import torch as T
from torchviz import make_dot
import matplotlib.pyplot as plt
from model_free.sac_x_hybrid.sac_x_hybrid_torch import Agent
import subprocess
from PyPDF2 import PdfMerger
from pdf2image import convert_from_path

# Inicializar o agente com parâmetros de exemplo
agent = Agent(alpha=0.0003, beta=0.0003, input_dims=10, max_action=1, 
              gamma=0.99, n_actions=2, max_size=100000, tau=0.001, batch_size=128,
              reward_scale=2, min_action=0, checkpoint_dir='tmp/sac_x', num_skills=3)

# Criar um estado de exemplo para passar pela rede
example_state = T.randn(1, 10).to(agent.actor.device)  # Garantir que o estado esteja no mesmo dispositivo que o modelo
example_action = T.randn(1, 2).to(agent.critic_1.device)  # Garantir que a ação esteja no mesmo dispositivo que o modelo

# Função para criar e salvar a visualização do grafo computacional
def create_graph(output, model, filename):
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'pdf'
    dot.render(filename)
    return filename + '.pdf'

# Função para cropar PDFs usando pdfcrop
def crop_pdf(input_pdf, output_pdf):
    subprocess.run(['pdfcrop', input_pdf, output_pdf])

# Gerar visualizações para cada rede
actor_output = agent.actor(example_state)[0]
critic1_output = agent.critic_1(example_state, example_action)
critic2_output = agent.critic_2(example_state, example_action)
value_output = agent.value(example_state)
target_value_output = agent.target_value(example_state)

# Criar e salvar os grafos computacionais como PDFs
actor_graph_path = create_graph(actor_output, agent.actor, "sac_x_actor_network")
critic1_graph_path = create_graph(critic1_output, agent.critic_1, "sac_x_critic1_network")
critic2_graph_path = create_graph(critic2_output, agent.critic_2, "sac_x_critic2_network")
value_graph_path = create_graph(value_output, agent.value, "sac_x_value_network")
target_value_graph_path = create_graph(target_value_output, agent.target_value, "sac_x_target_value_network")

# Cropar os PDFs gerados
crop_pdf(actor_graph_path, actor_graph_path)
crop_pdf(critic1_graph_path, critic1_graph_path)
crop_pdf(critic2_graph_path, critic2_graph_path)
crop_pdf(value_graph_path, value_graph_path)
crop_pdf(target_value_graph_path, target_value_graph_path)

# Função para carregar PDFs e convertê-los em imagens
def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    return images[0]

# Carregar PDFs como imagens
actor_image = pdf_to_image(actor_graph_path)
critic1_image = pdf_to_image(critic1_graph_path)
critic2_image = pdf_to_image(critic2_graph_path)
value_image = pdf_to_image(value_graph_path)
target_value_image = pdf_to_image(target_value_graph_path)

# Combinar todas as imagens em uma única figura
fig, axs = plt.subplots(2, 3, figsize=(25, 15))  # Alterado para 2x3 grid

# Adicionar títulos e imagens aos subplots
axs[0, 0].imshow(actor_image)
axs[0, 0].set_title('Actor Network')
axs[0, 0].axis('off')

axs[0, 1].imshow(critic1_image)
axs[0, 1].set_title('Critic Network 1')
axs[0, 1].axis('off')

axs[0, 2].imshow(critic2_image)
axs[0, 2].set_title('Critic Network 2')
axs[0, 2].axis('off')

axs[1, 0].imshow(value_image)
axs[1, 0].set_title('Value Network')
axs[1, 0].axis('off')

axs[1, 1].imshow(target_value_image)
axs[1, 1].set_title('Target Value Network')
axs[1, 1].axis('off')

# Remover subplot extra
fig.delaxes(axs[1, 2])

# Salvar a figura combinada como PDF
plt.savefig('sac_x_combined_networks.pdf')
plt.show()
