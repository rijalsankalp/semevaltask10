import os
import pandas as pd

class LoadData():
    """
    Class to load data from the given directory and subdirectories

    Args:
    base_dir (str): The base directory where the data is stored
    txt_file (str): The name of the text file containing the data
    subdirs (list): List of subdirectories to search for the text file
    
    Returns:
    data (DataFrame): DataFrame containing the loaded data
    """
    def load_data(self, base_dir, txt_file = "subtask-1-annotations.txt", subdir="EN", is_test = False):
        """
        Load data from the given directory and subdirectories
        
        Args:
        base_dir (str): The base directory where the data is stored
        txt_file (str): The name of the text file containing the data
        subdirs (list): List of subdirectories to search for the text file
        
        Returns:
        data (DataFrame): DataFrame containing the loaded data
        """
        # List to store DataFrames
        dataframes = []
        if not is_test:
            
            file_path = os.path.join(base_dir, subdir, txt_file)
            rows = []
            
            # Open file and process manually
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Skip empty or malformed lines
                    if not line.strip():
                        continue
                    
                    # Split the line on tabs
                    parts = line.strip().split('\t')
                    
                    # Ensure the line has at least 5 parts (article_id, entity_mention, start_offset, end_offset, main_role)
                    if len(parts) < 5:
                        continue
                    
                    # Extract required fields
                    article_id = parts[0]
                    entity_mention = parts[1]
                    start_offset = parts[2]
                    end_offset = parts[3]
                    main_role = parts[4]
                    
                    # Extract sub_roles
                    sub_roles = [role for role in parts[5:] if role.strip()]
                    
                    # Append the row
                    rows.append([article_id, entity_mention, start_offset, end_offset, main_role, sub_roles])
            
            # Create a DataFrame from the rows
            df = pd.DataFrame(rows, columns=['article_id', 'entity_mention', 'start_offset', 'end_offset', 'main_role', 'sub_roles'])
            dataframes.append(df)
        else:
            for subdir in subdirs:
                file_path = os.path.join(base_dir, subdir, txt_file)
                rows = []
                
                # Open file and process manually
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        # Skip empty or malformed lines
                        if not line.strip():
                            continue
                        
                        # Split the line on tabs
                        parts = line.strip().split('\t')
                        
                        # Ensure the line has at least 5 parts (article_id, entity_mention, start_offset, end_offset, main_role)
                        if len(parts) < 4:
                            continue
                        
                        # Extract required fields
                        article_id = parts[0]
                        entity_mention = parts[1]
                        start_offset = parts[2]
                        end_offset = parts[3]
                        
                        # Append the row
                        rows.append([article_id, entity_mention, start_offset, end_offset])
                
                # Create a DataFrame from the rows
                df = pd.DataFrame(rows, columns=['article_id', 'entity_mention', 'start_offset', 'end_offset'])
                dataframes.append(df)
        
        # Combine all DataFrames
        data = pd.concat(dataframes, ignore_index=True)
        # Shuffle data and reset index
        data = data.sample(frac=1).reset_index(drop=True)

        return data

if __name__ == "__main__":
    base_dir = "train"
    txt_file = "subtask-1-annotations.txt"
    subdirs = ['EN']

    data = LoadData()
    data = data.load_data(base_dir, txt_file, subdirs)
    print(data.head())

    data = LoadData()
    data = data.load_data("dev", "subtask-1-entity-mentions.txt", ["EN"], is_test = True)
    print(data.head())
    