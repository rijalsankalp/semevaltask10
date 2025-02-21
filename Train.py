import argparse
import os
import fasttext.util
import numpy as np
import pickle
from LoadData import LoadData
from EntityDataset import TrainDataset, TestDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class LanguageTrainer:
    def __init__(self, all_sub_roles: list, language="EN", random_seed=42):
        self.language = language
        np.random.seed(random_seed)
        self.ft_model = None
        self.main_classifier = None
        self.sub_role_classifiers = {}
        self.all_sub_roles = all_sub_roles
        
    def load_models(self):
        # Load fasttext model for the specific language
        if not os.path.exists(f'cc.{self.language.lower()}.300.bin'):
            fasttext.util.download_model(self.language.lower(), if_exists='ignore')
        self.ft_model = fasttext.load_model(f'cc.{self.language.lower()}.300.bin')
        
    def train(self, train_data_path):
        # Load data
        ldr = LoadData()
        train_data = ldr.load_data(base_dir=train_data_path, subdirs=[self.language])
        
        # Create dataset
        train_dataset = TrainDataset(dataframe=train_data, base_dir='train', language=self.language)
        
        # Prepare training data
        X_train, y_train = self._prepare_data(train_dataset)
        
        # Train main classifier
        self.main_classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.main_classifier.fit(X_train, y_train)

        # Save the main classifier
        os.makedirs(f'models/{self.language}', exist_ok=True)
        with open(f'models/{self.language}/main_classifier_{self.language}.pkl', 'wb') as f:
            pickle.dump(self.main_classifier, f)
        
        # Train sub-role classifiers
        self._train_sub_role_classifiers(train_dataset, X_train)

        # Save the sub-role classifiers
        for sub_role, classifier in self.sub_role_classifiers.items():
            with open(f'models/{self.language}/sub_role_classifier_{sub_role}_{self.language}.pkl', 'wb') as f:
                pickle.dump(classifier, f)
    
    def _prepare_data(self, dataset):
        X = []
        y = []
        for i in range(len(dataset)):
            item = dataset[i]
            if item is not None and item['word_features']:
                feature_vector = self.ft_model.get_sentence_vector(item['word_features'])
                X.append(feature_vector)
                y.append(item['main_role'])
        return np.array(X), np.array(y)
    
    # def _train_sub_role_classifiers(self, train_dataset, X_train):
    #     # Create binary labels for each sub-role
    #     sub_role_labels = {role: [] for role in self.all_sub_roles}
    #     X_train_replicated = []
        
    #     for i in range(len(train_dataset)):
    #         item = train_dataset[i]
    #         if item is not None and item['word_features']:
    #             feature_vector = self.ft_model.get_sentence_vector(item['word_features'])
    #             for sub_role in self.all_sub_roles:
    #                 sub_role_labels[sub_role].append(1 if sub_role in item['sub_roles'] else 0)
    #                 X_train_replicated.append(feature_vector)
        
    #     # Train a binary classifier for each sub-role
    #     for sub_role in self.all_sub_roles:
    #         classifier = LogisticRegression(random_state=42, max_iter=1000)
    #         y_sub = np.array(sub_role_labels[sub_role])
    #         classifier.fit(X_train_replicated, y_sub)
    #         self.sub_role_classifiers[sub_role] = classifier
    def _train_sub_role_classifiers(self, train_dataset, X_train):
        # Create binary labels for each sub-role
        sub_role_labels = {role: [] for role in self.all_sub_roles}
        X_train_replicated = []
    
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            if item is not None and item['word_features']:
                feature_vector = self.ft_model.get_sentence_vector(item['word_features'])
                X_train_replicated.append(feature_vector)  # Append only once per item
                for sub_role in self.all_sub_roles:
                    sub_role_labels[sub_role].append(1 if sub_role in item['sub_roles'] else 0)
    
        X_train_replicated = np.array(X_train_replicated)  # Convert to NumPy array
    
        # Train a binary classifier for each sub-role
        for sub_role in self.all_sub_roles:
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            y_sub = np.array(sub_role_labels[sub_role])
            
            if len(X_train_replicated) != len(y_sub):  # Debugging step
                print(f"Mismatch: X_train={len(X_train_replicated)}, y_sub={len(y_sub)}")
    
            classifier.fit(X_train_replicated, y_sub)
            self.sub_role_classifiers[sub_role] = classifier


class LanguageEvaluator:
    def __init__(self, language="EN"):
        self.language = language.lower()
        self.ft_model = None
        self.main_classifier = None
        self.sub_role_classifiers = {}
        
    def load_models(self):
        # Load fasttext model for the specific language
        self.ft_model = fasttext.load_model(f'cc.{self.language}.300.bin')
        
        # Load the main classifier
        with open(f'models/{self.language}/main_classifier_{self.language}.pkl', 'rb') as f:
            self.main_classifier = pickle.load(f)
        
        # Load the sub-role classifiers
        for sub_role in self.all_sub_roles:
            with open(f'models/{self.language}/sub_role_classifier_{sub_role}_{self.language}.pkl', 'rb') as f:
                self.sub_role_classifiers[sub_role] = pickle.load(f)
    
    def evaluate(self, test_data_path):
        # Load data
        ldr = LoadData()
        test_data = ldr.load_data(base_dir=test_data_path, subdirs=[self.language])
        
        # Create dataset
        test_dataset = TestDataset(dataframe=test_data, base_dir='test', language=self.language, folder="subtask-1-documents")
        
        # Prepare test data
        X_test, y_test = self._prepare_data(test_dataset)
        
        # Main role evaluation
        main_predictions = self.main_classifier.predict(X_test)
        main_accuracy = self.main_classifier.score(X_test, y_test)
        
        # Sub-role evaluation
        sub_role_results = self._evaluate_sub_roles(test_dataset, X_test)
        
        return {
            'main_accuracy': main_accuracy,
            'sub_role_results': sub_role_results
        }
    
    def _prepare_data(self, dataset):
        X = []
        y = []
        for i in range(len(dataset)):
            item = dataset[i]
            if item is not None and item['word_features']:
                feature_vector = self.ft_model.get_sentence_vector(item['word_features'])
                X.append(feature_vector)
                y.append(item['main_role'])
        return np.array(X), np.array(y)
    
    def _evaluate_sub_roles(self, test_dataset, X_test):
        y_test_sub = []
        y_pred_sub = []
        
        for i, feature_vector in enumerate(X_test):
            item = test_dataset[i]
            if item is not None and item['word_features']:
                y_test_sub.append(item['sub_roles'])
                predictions = self._get_predictions(feature_vector)
                y_pred_sub.append(predictions)
        
        emr = sum(1 for true, pred in zip(y_test_sub, y_pred_sub) 
                 if set(true) == set(pred)) / len(y_test_sub)
        
        return {'exact_match_ratio': emr}

    def _get_predictions(self, feature_vector):
        sub_role_pred = {}
        for sub_role in self.all_sub_roles:
            prob = self.sub_role_classifiers[sub_role].predict_proba([feature_vector])[0][1]
            sub_role_pred[sub_role] = prob
            
        sorted_pred = dict(sorted(sub_role_pred.items(), key=lambda item: item[1], reverse=True))
        
        predicted_sub_roles = []
        for key, value in sorted_pred.items():
            if not predicted_sub_roles or value >= 0.5 * list(sorted_pred.values())[0]:
                predicted_sub_roles.append(key)
            else:
                break
                
        return predicted_sub_roles

def main():
    # all_main_roles = ["Protagonist", "Antagonist", "Innocent"]
    all_sub_roles = ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous", "Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot", "Forgotten", "Exploited", "Victim", "Scapegoat"]
    # sub_dict = {
    #     "Protagonist":["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
    #     "Innocent":["Forgotten", "Exploited", "Victim", "Scapegoat"],
    #     "Antagonist":["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"]
    # }
    parser = argparse.ArgumentParser(description='Train and evaluate role classification model for different languages')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], help='Mode: train or evaluate')
    parser.add_argument('--language', type=str, default='EN', help='Language code (en, ru, hi, etc.)')
    parser.add_argument('--train_path', type=str, default = "train", help='Path to training data')
    parser.add_argument('--test_path', type=str, default = "test", help='Path to test data')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        trainer = LanguageTrainer(language=args.language, all_sub_roles=all_sub_roles)
        trainer.load_models()
        trainer.train(args.train_path)
    elif args.mode == 'evaluate':
        evaluator = LanguageEvaluator(language=args.language)
        evaluator.load_models()
        results = evaluator.evaluate(args.test_path)
        
        print(f"Results for language {args.language}:")
        print(f"Main role accuracy: {results['main_accuracy']:.4f}")
        print(f"Sub-role exact match ratio: {results['sub_role_results']['exact_match_ratio']:.4f}")

if __name__ == "__main__":
    main()
