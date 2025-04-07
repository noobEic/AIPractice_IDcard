import docx2txt
import os
import textract
def convert_doc_to_txt(doc_file_path):
    text = docx2txt.process(doc_file_path)
    txt_file_path = doc_file_path.replace('.docx', '.txt')
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(text)
if __name__ == '__main__':
    for file in os.listdir('.'):
        if file.endswith('.docx'):
            try:
                convert_doc_to_txt(file)
            except:
                pass
        if file.endswith('.doc'):
            
            text = textract.process(file)
            txt_file_path = file.replace('.doc', '.txt')
            with open(txt_file_path, 'wb') as f:
                f.write(text)
           