# Install if needed
!pip install ipywidgets

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')

class IMDB_Sentiment_Analyzer_Web:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.accuracy = 0
        self.df = None
        
        # Create widgets
        self.setup_widgets()
        
    def setup_widgets(self):
        # Title
        self.title = widgets.HTML(value="<h1 style='color: #2c3e50;'>🎬 IMDB Movie Review Sentiment Analyzer</h1>")
        
        # Data loading section
        self.load_button = widgets.Button(
            description="📁 Load & Process Data",
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.load_button.on_click(self.load_data)
        
        self.load_status = widgets.HTML(value="<i>Click 'Load & Process Data' to begin</i>")
        
        # Model training section
        self.features_slider = widgets.IntSlider(
            value=5000,
            min=1000,
            max=10000,
            step=1000,
            description='Max Features:',
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        
        self.train_button = widgets.Button(
            description="🤖 Train Model",
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.train_button.on_click(self.train_model)
        
        self.train_status = widgets.HTML(value="")
        self.model_info = widgets.HTML(value="<span style='color: #e74c3c;'>Model not trained</span>")
        
        # Review analysis section
        self.review_text = widgets.Textarea(
            value="",
            placeholder="Enter a movie review here...",
            layout=widgets.Layout(width='95%', height='150px')
        )
        
        # Sample reviews
        self.sample_dropdown = widgets.Dropdown(
            options=[('Positive Sample', 'positive'), ('Negative Sample', 'negative')],
            value='positive',
            description='Sample:'
        )
        self.sample_dropdown.observe(self.load_sample, names='value')
        
        self.analyze_button = widgets.Button(
            description="📝 Analyze Sentiment",
            button_style='info',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.analyze_button.on_click(self.analyze_sentiment)
        
        # Results display
        self.result_output = widgets.Output()
        
        # Accuracy display
        self.accuracy_display = widgets.HTML(
            value="<h2 style='background-color: #34495e; color: white; padding: 10px; border-radius: 5px;'>Accuracy: N/A</h2>"
        )
        
        # Visualization output
        self.viz_output = widgets.Output()
        
        # Data info
        self.data_info = widgets.HTML(value="<i>Data not loaded</i>")
        
        # Status bar
        self.status_bar = widgets.HTML(value="<div style='background-color: #2c3e50; color: white; padding: 5px;'>Ready</div>")
        
        # Arrange layout
        self.ui = widgets.VBox([
            self.title,
            widgets.HBox([
                widgets.VBox([
                    widgets.VBox([
                        widgets.HTML(value="<h3>📁 Data Loading</h3>"),
                        self.load_button,
                        self.load_status,
                        widgets.HTML(value="<hr>"),
                        widgets.HTML(value="<h3>🤖 Model Training</h3>"),
                        self.features_slider,
                        self.train_button,
                        self.train_status,
                        self.model_info,
                        widgets.HTML(value="<hr>"),
                        widgets.HTML(value="<h3>📝 Analyze Your Review</h3>"),
                        self.review_text,
                        self.sample_dropdown,
                        self.analyze_button,
                        self.result_output,
                    ], layout=widgets.Layout(width='50%', border='solid 1px #ddd', padding='10px')),
                ]),
                widgets.VBox([
                    self.accuracy_display,
                    widgets.HTML(value="<h3>📊 Model Performance</h3>"),
                    self.viz_output,
                    widgets.HTML(value="<h3>📋 Dataset Information</h3>"),
                    self.data_info,
                ], layout=widgets.Layout(width='50%', border='solid 1px #ddd', padding='10px')),
            ]),
            self.status_bar
        ])
    
    def load_data(self, b):
        """Load and process the IMDB dataset"""
        with self.result_output:
            clear_output()
            print("Loading dataset...")
        
        try:
            # Update status
            self.load_status.value = "<span style='color: #f39c12;'>Loading...</span>"
            self.load_button.disabled = True
            self.status_bar.value = "<div style='background-color: #f39c12; color: white; padding: 5px;'>Loading dataset...</div>"
            
            # Load dataset
            self.df = pd.read_csv('/content/IMDB Dataset.csv')
            
            # Convert sentiment to binary
            self.df['sentiment_binary'] = self.df['sentiment'].map({'positive': 1, 'negative': 0})
            
            # Update data info
            pos_count = (self.df['sentiment_binary'] == 1).sum()
            neg_count = (self.df['sentiment_binary'] == 0).sum()
            
            info_text = f"""
            <div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>
            <b>✅ Dataset loaded successfully!</b><br>
            • Total reviews: {len(self.df):,}<br>
            • Positive reviews: {pos_count:,}<br>
            • Negative reviews: {neg_count:,}<br>
            • Positive/Negative ratio: {pos_count/neg_count:.2f}:1
            </div>
            """
            
            self.data_info.value = info_text
            self.load_status.value = "<span style='color: #27ae60;'>✅ Dataset loaded!</span>"
            self.status_bar.value = "<div style='background-color: #27ae60; color: white; padding: 5px;'>Dataset loaded successfully!</div>"
            
            # Enable train button
            self.train_button.disabled = False
            
            with self.result_output:
                clear_output()
                print(f"✅ Dataset loaded with {len(self.df):,} reviews")
                print(f"   - Positive: {pos_count:,}")
                print(f"   - Negative: {neg_count:,}")
                
        except Exception as e:
            self.load_status.value = f"<span style='color: #e74c3c;'>❌ Load failed: {str(e)}</span>"
            self.status_bar.value = f"<div style='background-color: #e74c3c; color: white; padding: 5px;'>Error loading dataset</div>"
            
            with self.result_output:
                clear_output()
                print(f"❌ Error loading dataset: {e}")
        finally:
            self.load_button.disabled = False
    
    def train_model(self, b):
        """Train the sentiment analysis model"""
        if self.df is None:
            with self.result_output:
                clear_output()
                print("❌ Please load the dataset first!")
            return
        
        with self.result_output:
            clear_output()
            print("Training model...")
        
        try:
            # Update status
            self.train_status.value = "<span style='color: #f39c12;'>Training model...</span>"
            self.model_info.value = "<span style='color: #f39c12;'>Training in progress...</span>"
            self.train_button.disabled = True
            self.status_bar.value = "<div style='background-color: #f39c12; color: white; padding: 5px;'>Training model...</div>"
            
            # Get parameters
            max_features = self.features_slider.value
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.df['review'], self.df['sentiment_binary'], 
                test_size=0.2, random_state=42
            )
            
            # Create TF-IDF features
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_vec)
            self.accuracy = accuracy_score(y_test, y_pred)
            
            # Update UI
            self.accuracy_display.value = f"""
            <h2 style='background-color: #34495e; color: white; padding: 10px; border-radius: 5px;'>
            Accuracy: {self.accuracy:.2%}
            </h2>
            """
            
            vocab_size = len(self.vectorizer.get_feature_names_out())
            self.model_info.value = f"""
            <div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>
            <b>✅ Model trained successfully!</b><br>
            • Features used: {vocab_size:,}<br>
            • Max iterations: 1000<br>
            • Test set size: {len(X_test):,} reviews
            </div>
            """
            
            self.train_status.value = "<span style='color: #27ae60;'>✅ Training complete!</span>"
            self.status_bar.value = f"<div style='background-color: #27ae60; color: white; padding: 5px;'>Model trained with accuracy: {self.accuracy:.2%}</div>"
            
            # Update visualization
            self.update_visualization(y_test, y_pred)
            
            # Enable analyze button
            self.analyze_button.disabled = False
            
            with self.result_output:
                clear_output()
                print(f"✅ Model trained successfully!")
                print(f"   - Accuracy: {self.accuracy:.2%}")
                print(f"   - Features: {vocab_size:,}")
                print(f"   - Test samples: {len(X_test):,}")
                
        except Exception as e:
            self.train_status.value = f"<span style='color: #e74c3c;'>❌ Training failed: {str(e)}</span>"
            self.model_info.value = f"<span style='color: #e74c3c;'>Model training failed</span>"
            self.status_bar.value = f"<div style='background-color: #e74c3c; color: white; padding: 5px;'>Training failed</div>"
            
            with self.result_output:
                clear_output()
                print(f"❌ Training error: {e}")
        finally:
            self.train_button.disabled = False
    
    def update_visualization(self, y_test, y_pred):
        """Update the performance visualization"""
        with self.viz_output:
            clear_output()
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot 1: Accuracy bar
            ax1.bar(['Model Accuracy'], [self.accuracy], color=['#3498db'], alpha=0.7)
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'Model Performance: {self.accuracy:.2%}')
            ax1.legend()
            
            # Plot 2: Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax2.set_title('Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
            plt.tight_layout()
            plt.show()
    
    def load_sample(self, change):
        """Load sample review based on selection"""
        if change['type'] == 'change' and change['name'] == 'value':
            sample_type = change['new']
            
            if sample_type == "positive":
                sample_text = "This movie was absolutely fantastic! The acting was superb, the storyline was engaging from beginning to end, and the cinematography was breathtaking. I would definitely recommend this to everyone."
            else:  # negative
                sample_text = "I was very disappointed with this film. The plot was predictable, the characters were poorly developed, and the dialogue felt forced. I wouldn't recommend wasting your time on this one."
            
            self.review_text.value = sample_text
    
    def analyze_sentiment(self, b):
        """Analyze sentiment of user input"""
        if self.model is None:
            with self.result_output:
                clear_output()
                print("❌ Please train the model first!")
            return
        
        review = self.review_text.value.strip()
        
        if not review:
            with self.result_output:
                clear_output()
                print("❌ Please enter a review to analyze!")
            return
        
        try:
            with self.result_output:
                clear_output()
                print("Analyzing sentiment...")
            
            # Vectorize the review
            review_vec = self.vectorizer.transform([review])
            
            # Predict
            prediction = self.model.predict(review_vec)[0]
            probability = self.model.predict_proba(review_vec)[0]
            
            # Get confidence
            confidence = probability[prediction] * 100
            
            # Update result display
            sentiment = "Positive" if prediction == 1 else "Negative"
            color = "#27ae60" if prediction == 1 else "#e74c3c"
            
            with self.result_output:
                clear_output()
                print(f"📊 Analysis Results:")
                print(f"   Sentiment: {sentiment}")
                print(f"   Confidence: {confidence:.1f}%")
                print(f"   Probability breakdown:")
                print(f"     - Positive: {probability[1]*100:.1f}%")
                print(f"     - Negative: {probability[0]*100:.1f}%")
                print("\n💭 Review preview:")
                print(f"   '{review[:100]}...'")
            
            self.status_bar.value = f"<div style='background-color: {color}; color: white; padding: 5px;'>{sentiment} sentiment detected with {confidence:.1f}% confidence</div>"
            
        except Exception as e:
            with self.result_output:
                clear_output()
                print(f"❌ Analysis error: {e}")
    
    def show(self):
        """Display the UI"""
        display(self.ui)

# Create and display the application
app = IMDB_Sentiment_Analyzer_Web()
app.show()
