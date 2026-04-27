"""
Vision Transformer (ViT) Plant Disease Detection
Replaces CNN with Vision Transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.projection(out)
        
        return out

class TransformerBlock(nn.Module):
    """Vision Transformer Block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.dropout(self.attention(self.ln1(x)))
        # MLP with residual connection
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for Plant Disease Detection"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, num_heads=12, num_layers=12, 
                 num_classes=38, dropout=0.1):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.patch_embedding.n_patches + 1, embed_dim)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classification
        x = self.ln(x)
        cls_token_final = x[:, 0]  # Use CLS token for classification
        x = self.head(cls_token_final)
        
        return x

class ViTPlantDiseaseDetector:
    """Vision Transformer Plant Disease Detection System"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Plant disease classes
        self.classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
            'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew',
            'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
            'Corn___Northern_Leaf_Blight', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___healthy',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
            'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
            'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
        ]
        
        # Disease information database
        self.disease_info = {
            'healthy': {
                'treatment': 'Your plant appears healthy! Continue with regular care including proper watering, sunlight, and monitoring for any changes.',
                'severity': 'Low',
                'action': 'Preventive care'
            },
            'scab': {
                'treatment': 'Apply fungicide containing copper or sulfur. Remove affected leaves and improve air circulation.',
                'severity': 'Medium',
                'action': 'Fungicide treatment'
            },
            'rust': {
                'treatment': 'Remove infected parts, apply fungicide, and ensure proper spacing between plants.',
                'severity': 'Medium',
                'action': 'Fungicide application'
            },
            'blight': {
                'treatment': 'Apply appropriate fungicide, remove affected plants, and practice crop rotation.',
                'severity': 'High',
                'action': 'Immediate treatment required'
            },
            'spot': {
                'treatment': 'Remove spotted leaves, apply copper-based fungicide, and ensure proper drainage.',
                'severity': 'Medium',
                'action': 'Targeted treatment'
            },
            'mold': {
                'treatment': 'Improve ventilation, reduce humidity, apply fungicide, and remove affected parts.',
                'severity': 'High',
                'action': 'Environmental control + treatment'
            },
            'virus': {
                'treatment': 'Remove infected plants, control insect vectors, and use virus-resistant varieties.',
                'severity': 'High',
                'action': 'Plant removal + prevention'
            },
            'bacterial': {
                'treatment': 'Apply copper-based bactericide, remove infected parts, and avoid overhead watering.',
                'severity': 'Medium',
                'action': 'Bactericide treatment'
            }
        }
        
    def load_model(self, model_path='vit_plant_disease_model.pth'):
        """Load pre-trained ViT model"""
        try:
            self.model = VisionTransformer(
                img_size=224, patch_size=16, in_channels=3,
                embed_dim=768, num_heads=12, num_layers=12,
                num_classes=len(self.classes), dropout=0.1
            ).to(self.device)
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ ViT model loaded from {model_path}")
            else:
                print("⚠️ No pre-trained model found, using randomly initialized weights")
                # Initialize with proper weights for demo
                self._initialize_demo_weights()
                
            self.model.eval()
            return True
            
        except Exception as e:
            print(f"❌ Error loading ViT model: {e}")
            return False
    
    def _initialize_demo_weights(self):
        """Initialize model with demo weights"""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def predict(self, image_path):
        """Predict plant disease using Vision Transformer"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get top 3 predictions
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                
            # Process results
            disease_class = self.classes[predicted.item()]
            confidence_score = confidence.item() * 100
            
            # Get disease type and treatment
            disease_type = self._get_disease_type(disease_class)
            treatment = self._get_treatment(disease_class)
            
            # Format results
            results = {
                'disease': disease_class.replace('___', ' - ').replace('_', ' '),
                'confidence': round(confidence_score, 2),
                'top_predictions': [
                    {
                        'disease': self.classes[idx].replace('___', ' - ').replace('_', ' '),
                        'confidence': round(prob.item() * 100, 2)
                    }
                    for idx, prob in zip(top3_idx[0], top3_prob[0])
                ],
                'treatment': treatment,
                'disease_type': disease_type,
                'model_type': 'Vision Transformer (ViT)',
                'attention_info': 'Using self-attention mechanism to analyze plant features'
            }
            
            return results
            
        except Exception as e:
            return {
                'error': f'ViT prediction failed: {str(e)}',
                'disease': 'Unknown',
                'confidence': 0,
                'treatment': 'Please try with a clearer image.',
                'model_type': 'Vision Transformer (ViT)'
            }
    
    def _get_disease_type(self, disease_class):
        """Categorize disease type"""
        disease_lower = disease_class.lower()
        
        if 'healthy' in disease_lower:
            return 'healthy'
        elif any(keyword in disease_lower for keyword in ['scab', 'rot', 'rust', 'mold']):
            return 'fungal'
        elif any(keyword in disease_lower for keyword in ['bacterial', 'spot']):
            return 'bacterial'
        elif any(keyword in disease_lower for keyword in ['virus', 'mosaic']):
            return 'viral'
        elif 'blight' in disease_lower:
            return 'fungal'
        else:
            return 'other'
    
    def _get_treatment(self, disease_class):
        """Get treatment recommendation"""
        disease_type = self._get_disease_type(disease_class)
        return self.disease_info.get(disease_type, self.disease_info['healthy'])

# Initialize ViT detector
vit_detector = ViTPlantDiseaseDetector()

def analyze_plant_image_vit(image_path):
    """Main function for plant disease analysis using Vision Transformer"""
    print("🔬 Initializing Vision Transformer (ViT) for Plant Disease Detection...")
    
    # Load ViT model
    if not vit_detector.load_model():
        return {
            'error': 'Failed to load ViT model',
            'disease': 'Model Error',
            'confidence': 0,
            'treatment': 'Please check model installation.',
            'model_type': 'Vision Transformer (ViT)'
        }
    
    print(f"📸 Analyzing image: {image_path}")
    print("🧠 Using Vision Transformer with self-attention mechanism...")
    
    # Perform prediction
    results = vit_detector.predict(image_path)
    
    print("✅ ViT Analysis Complete!")
    print(f"🔍 Disease: {results.get('disease', 'Unknown')}")
    print(f"📊 Confidence: {results.get('confidence', 0)}%")
    print(f"🤖 Model: {results.get('model_type', 'Vision Transformer')}")
    
    return results

if __name__ == "__main__":
    # Demo usage
    print("🌱 Vision Transformer Plant Disease Detection System")
    print("=" * 50)
    
    # Test with sample image (if available)
    sample_image = "test_plant.jpg"
    if os.path.exists(sample_image):
        results = analyze_plant_image_vit(sample_image)
        print(f"\n📋 Results: {json.dumps(results, indent=2)}")
    else:
        print("📸 Please provide an image file for analysis")
        print("💡 Usage: python vit_plant_disease.py <image_path>")
