import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

class EdgeEncoder(MessagePassing):
    def __init__(self, in_channels, edge_dim, hidden_dim):
        super(EdgeEncoder, self).__init__(aggr='add')
        self.node_mlp = torch.nn.Linear(in_channels + hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.LeakyReLU(0.15),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_mlp(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, edge_attr], dim=1)
        return self.node_mlp(z)

    def update(self, aggr_out):
        return aggr_out

class EdgeVGAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, latent_dim):
        super(EdgeVGAEEncoder, self).__init__()
        self.conv1 = EdgeEncoder(input_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeEncoder(hidden_dim, edge_dim, hidden_dim)
        self.drop = torch.nn.Dropout(0.05)
        self.mu_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.drop(x)
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr), 0.15)
        x = self.drop(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr), 0.15)
        return self.mu_layer(x), self.logvar_layer(x)

class EdgeVGAE(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, latent_dim, num_classes):
        super(EdgeVGAE, self).__init__()
        self.encoder = EdgeVGAEEncoder(input_dim, edge_dim, hidden_dim, latent_dim)
        self.classifier = torch.nn.Linear(latent_dim, num_classes)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, latent_dim),
            torch.nn.LeakyReLU(0.15),
            torch.nn.Linear(latent_dim, edge_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.15)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps_val = torch.randn_like(std)
        return mu + eps_val * std

    def forward(self, x, edge_index, edge_attr, batch, eps=1.0):
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        if eps == 0.0:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        
        
        graph_z = global_mean_pool(z, batch)
        class_logits = self.classifier(graph_z)
        return z, mu, logvar, class_logits

    def decode(self, z, edge_index): 
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        
        row, col = edge_index
        edge_node_features = torch.cat([z[row], z[col]], dim=-1)
        edge_attr_pred = self.edge_mlp(edge_node_features)
        edge_attr_pred = torch.sigmoid(edge_attr_pred)
        return adj_pred, edge_attr_pred

    def recon_loss(self, z, edge_index, edge_attr):
        adj_pred, edge_attr_pred = self.decode(z, edge_index)
        
        adj_true = torch.zeros_like(adj_pred, dtype=torch.float32)
        adj_true[edge_index[0], edge_index[1]] = 1.0
        adj_loss = F.binary_cross_entropy(adj_pred, adj_true, reduction='sum') / z.size(0)

        edge_loss = F.mse_loss(edge_attr_pred, edge_attr)

        return 0.1 * adj_loss + edge_loss

    def denoise_recon_loss(self, z, edge_index):
        adj_recon, _ = self.decode(z, edge_index)
        adj_clean = torch.zeros_like(adj_recon, dtype=torch.float32)
        adj_clean[edge_index[0], edge_index[1]] = 1.0
        loss = F.mse_loss(adj_recon, adj_clean, reduction='sum') / z.size(0)
        return loss

    def kl_loss(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))