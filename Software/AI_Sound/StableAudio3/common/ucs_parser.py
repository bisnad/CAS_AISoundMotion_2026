import csv
import os
import torchaudio

ucs_definition_file_path = "E:/Data/audio/UCS/Meta/UCS_v8.2.1.csv"

def load_ucs_definition_file(file_path):
    
    ucs_definitons = []
    ucs_labels = None
    
    label_row_index = 2
    
    with open(file_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        
        for i, row in enumerate(reader):
            
            if i == label_row_index:
                
                ucs_labels = row
                
            elif i > label_row_index:
            
                usc_definition = {}
                
                for j in range(len(row)):
                    
                    usc_definition[ucs_labels[j]] = row[j].lower()
                    
                ucs_definitons.append(usc_definition)
                
    return ucs_definitons

def gather_audio_files(dir_path):
    
    file_names = [ name for name in os.listdir(dir_path) ]
    
    return file_names
                    
def load_ucs_waveform_and_prompt(dir_path, file_name, ucs_definitions):
    
    waveform, sr = torchaudio.load(dir_path + "/" + file_name)
    
    #print("waveform s ", waveform.shape)
    
    prompt = None
    
    file_name_comma_separated = file_name.split(", ")
    file_name_underscore_separated = file_name.split("_")
    
    if len(file_name_comma_separated) >= 2:
        category_name = file_name_comma_separated[0].lower()
        sub_category_name = file_name_comma_separated[1].lower()
        
        #print("category_name ", category_name, " sub_category_name ", sub_category_name)
        
        prompt = category_name + ", " + sub_category_name 
        
        # search ucs definition file
        
        for ucs_definition in ucs_definitions:
            
            if ucs_definition["Category"] == category_name and ucs_definition["SubCategory"] == sub_category_name:

                prompt += ", " + ucs_definition["Synonyms - Comma Separated"] 
        
    else:
        cat_id_name = file_name_underscore_separated[0].split("-")[0].lower()
        
        #print("cat_id_name ", cat_id_name)
        
        # search ucs definition file
        for ucs_definition in ucs_definitions:
            if ucs_definition["CatID"] == cat_id_name:
                prompt = ucs_definition["Category"] + ", " + ucs_definition["SubCategory"] 
                prompt += ", " + ucs_definition["Synonyms - Comma Separated"] 
    
    return waveform, prompt
                    

ucs_definitons = load_ucs_definition_file(ucs_definition_file_path)


"""

audio_file_dir = "E:/Data/audio/UCS/Audio/48khz"

audio_file_names = gather_audio_files(audio_file_dir)

for audio_file_name in audio_file_names:
    
    print("audio_file_name ", audio_file_name)

    waveform, prompt = load_ucs_waveform_and_prompt(audio_file_dir, audio_file_name, ucs_definitons)

    print("waveform s ", waveform.shape)
    print("prompt ", prompt)

waveform, prompt = load_ucs_waveform_and_prompt(audio_file_dir, audio_file_names[0], ucs_definitons)



audio_file_names[0]

prompt


ucs_definitons[10]["Category"]
"""