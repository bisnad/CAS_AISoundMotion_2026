import numpy as np
import torch

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

config = {
    "model_encoder": None,
    "device": "cuda",
    "pose_sequence": None,
    "pose_sequence_length": 64,
    "pose_excerpt_offset": 20,
    "n_neighbors": 4
}

class MotionMapping():
    def __init__(self, config):
        self.encoder = config["model_encoder"]
        self.device = config["device"]
        self.pose_sequence = config["pose_sequence"]
        self.pose_dim = self.pose_sequence.shape[1]
        self.pose_sequence_length = config["pose_sequence_length"]
        self.pose_excerpt_offset = config["pose_excerpt_offset"]
        self.n_neighbors = config["n_neighbors"]
        self.motion_encodings = None
        self.Z_tsne = None
        
        self.generate_mapping()
        self.generate_neighbors()

    def generate_mapping(self):
        #create motion excerpts for 2D Mapping
        motion_excerpt_start_frame = 0
        motion_excerpt_end_frame = self.pose_sequence.shape[0]

        motion_excerpts = []
        for fI in range(motion_excerpt_start_frame, motion_excerpt_end_frame - self.pose_sequence_length, self.pose_excerpt_offset):
            motion_excerpt = self.pose_sequence[fI:fI + self.pose_sequence_length]
            motion_excerpts.append(motion_excerpt)
            
        motion_excerpts = np.stack(motion_excerpts, axis=0)
        motion_excerpts = torch.from_numpy(motion_excerpts).to(torch.float32)
        motion_excerpts = motion_excerpts.reshape(-1, self.pose_sequence_length, self.pose_dim)
        # Permute for CNN: [batch_size, channels, seq_len]
        motion_excerpts = motion_excerpts.permute(0, 2, 1) 

        # Generate latent encodings and 2D projections
        batch_size = 32

        with torch.no_grad():
            motion_encodings = []
            
            for eI in range(0, motion_excerpts.shape[0] - batch_size, batch_size):
                motion_encoder_in = motion_excerpts[eI:eI+batch_size].to(self.device)
                
                # new CNN encoder output shape is (mu, logvar)
                h = self.encoder(motion_encoder_in)
                mu, logvar = torch.chunk(h, 2, dim=1)
                std = torch.exp(0.5 * logvar)
                encoded_batch = mu + std * torch.randn_like(std)

                print("encoded_batch s ", encoded_batch.shape)
                
                motion_encodings.append(encoded_batch.detach().cpu())

            self.motion_encodings = torch.cat(motion_encodings, dim=0).numpy()
            
            encodings_flat = self.motion_encodings.reshape(self.motion_encodings.shape[0], -1)

            tsne = TSNE(n_components=2, max_iter=5000, verbose=1)
            self.Z_tsne = tsne.fit_transform(encodings_flat)

    def generate_neighbors(self):
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto', metric='euclidean')
        self.knn.fit(self.Z_tsne)

    def calc_distance_based_averaged_encoding(self, point2D):
        """Returns distance-weighted averaged encoding for a given 2D point."""
        _, indices = self.knn.kneighbors(point2D)
        nearest_positions = self.Z_tsne[indices[0]]
        
        # nearest_encodings will have shape [n_neighbors, latent_dim, seq_len]
        nearest_encodings = self.motion_encodings[indices[0]]
        
        nearest_2D_distances = np.linalg.norm(nearest_positions - point2D, axis=1)
        max_2D_distance = np.max(nearest_2D_distances)
        
        if max_2D_distance == 0.0:
            weights = np.ones_like(nearest_2D_distances)
        else:
            norm_nearest_2D_distances = nearest_2D_distances / max_2D_distance
            weights = (1.0 - norm_nearest_2D_distances)
            
        # Averages across the n_neighbors (axis=0), returning shape [latent_dim, seq_len]
        return np.average(nearest_encodings, weights=weights, axis=0)
