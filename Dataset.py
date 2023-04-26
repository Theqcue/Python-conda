# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:42:18 2023

@author: Helen
"""

import pandas as pd

#df = pd.read_csv('arxiv_data.csv')

#df_d = df['summaries'].str.contains("2023")

#df_big = pd.read_json('arxiv-metadata-oai-snapshot.json', lines=True, nrows=500)


#Please generate a research paper abstract based on this title: The abstract should briefly summarize the main focus of the paper and highlight any key findings or conclusions.

import openai

openai.api_key = "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"

df_big = pd.read_json('arxiv-metadata-oai-snapshot.json', lines=True, nrows=2000)

split_size = len(df_big) // 5

# Split the dataset into 5 smaller DataFrames
df1 = df_big.iloc[:split_size]
df2 = df_big.iloc[split_size:split_size*2]
df3 = df_big.iloc[split_size*2:split_size*3]
df4 = df_big.iloc[split_size*3:split_size*4]
df5 = df_big.iloc[split_size*4:]

##ÆNDRE DF
Currect_df = df1
result = []
i = 0
##ÆNDRE NAVN df_big til df_#





import random

for _, row in Currect_df.iterrows():
    
    title = row['title']

    prompts = [
        f"Please provide a brief abstract for the research paper with the title: {title}. Make sure to include the main points and any significant findings or conclusions?",
        f"Can you create a concise abstract for a research paper titled: {title}? Summarize the primary focus and highlight important outcomes or conclusions?",
        f"Based on the title: {title}, could you compose a summary that encompasses the main ideas and key results or conclusions of the research paper?",
        f"Write a short abstract for a research paper having the title: {title}. Briefly describe the main focus and mention any crucial findings or conclusions?",
        f"Construct a succinct abstract for the research paper with the title: {title}, emphasizing the core subject and any notable outcomes or conclusions?",
        f"For the research paper titled: {title}, please create a summary that briefly covers the central theme and highlights the most important findings or conclusions?",
        f"Please draft an abstract for the research paper with the title: {title} that outlines the main objectives and underlines key results or conclusions?",
        f"Develop a brief abstract for a research paper titled: {title} that captures the primary focus and important discoveries or conclusions?",
        f"Generate a compact abstract for the research paper titled: {title}, including the main points and any significant findings or conclusions?",
        f"Compose a research paper abstract based on the title: {title} that quickly summarizes the main focus and underlines any critical outcomes or conclusions?",
        f"Could you produce a concise abstract for the research paper titled: {title}, highlighting the main points and any essential findings or conclusions?",
        f"Please write a brief abstract for the research paper with the title: {title}, focusing on the main theme and any important results or conclusions?",
        f"Create a summary of the research paper titled: {title}, emphasizing the main subject and key findings or conclusions?",
        f"Craft a short abstract for the research paper with the title: {title}, outlining the central focus and any significant outcomes or conclusions?",
        f"Formulate a concise abstract for the research paper titled: {title}, encompassing the main ideas and important discoveries or conclusions?",
        f"Please generate an abstract for the research paper with the title: {title}, briefly describing the main objectives and highlighting any key findings or conclusions?",
        f"Can you draft a summary for the research paper titled: {title} that captures the main focus and essential results or conclusions?",
        f"Write a compact abstract for the research paper with the title: {title}, including the primary focus and any noteworthy findings or conclusions?",
        f"Develop a research paper abstract based on the title: {title} that briefly summarizes the main theme and underlines critical outcomes or conclusions?",
        f"Provide an abstract for the research paper titled: {title}, capturing the main ideas and any significant results or conclusions?",
        f"Prepare a succinct abstract for the research paper with the title: {title} that highlights the central focus and key findings or conclusions?",
        f"Create a research paper abstract for the title: {title} that briefly covers the main subject and any important outcomes or conclusions?",
        f"Write a summary of the research paper titled: {title}, emphasizing the main objectives and notable discoveries or conclusions?",
        f"Draft a brief abstract for the research paper with the title: {title}, outlining the primary theme and any essential findings or conclusions?",
        f"Compose an abstract for the research paper titled: {title} that captures the main focus and important results or conclusions?",
        f"Generate a research paper abstract based on the title: {title} that quickly summarizes the central subject and highlights any critical outcomes or conclusions?",
        f"Please provide a concise abstract for the research paper with the title: {title}, including the main points and any noteworthy findings or conclusions?",
        f"Can you create a short abstract for the research paper titled: {title} that emphasizes the main ideas and key results or conclusions?",
        f"Based on the title: {title}, could you write a summary that outlines the main focus and highlights any significant outcomes or conclusions?",
        f"Develop a compact abstract for the research paper with the title: {title}, focusing on the primary theme and any important findings or conclusions?",
        f"Please craft a concise abstract for the research paper titled: {title}, which covers the main points and any vital findings or conclusions?",
        f"Write a brief abstract for the research paper with the title: {title}, concentrating on the main theme and any noteworthy results or conclusions?",
        f"Compose a summary for the research paper titled: {title}, stressing the main subject and essential findings or conclusions?",
        f"Create a brief abstract for the research paper with the title: {title}, summarizing the central focus and any important outcomes or conclusions?",
        f"Design a concise abstract for the research paper titled: {title}, which encompasses the main ideas and significant discoveries or conclusions?",
        f"Generate an abstract for the research paper with the title: {title} that briefly outlines the main objectives and emphasizes key findings or conclusions?",
        f"Can you write a summary for the research paper titled: {title} that focuses on the main theme and crucial results or conclusions?",
        f"Produce a compact abstract for the research paper with the title: {title}, mentioning the primary focus and any remarkable findings or conclusions?",
        f"Create a research paper abstract based on the title: {title} that concisely summarizes the main theme and highlights essential outcomes or conclusions?",
        f"Give an abstract for the research paper titled: {title}, which captures the main ideas and any important results or conclusions?",
        f"Construct a succinct abstract for the research paper with the title: {title} that emphasizes the central focus and key findings or conclusions?",
        f"Develop a research paper abstract for the title: {title} that briefly addresses the main subject and any noteworthy outcomes or conclusions?",
        f"Provide a summary of the research paper titled: {title}, focusing on the main objectives and significant discoveries or conclusions?",
        f"Form a brief abstract for the research paper with the title: {title}, outlining the primary theme and any indispensable findings or conclusions?",
        f"Make an abstract for the research paper titled: {title} that conveys the main focus and relevant results or conclusions?",
        f"Generate a research paper abstract based on the title: {title} that rapidly summarizes the central subject and underlines any important outcomes or conclusions?",
        f"Please offer a concise abstract for the research paper with the title: {title}, comprising the main points and any remarkable findings or conclusions?",
        f"Can you construct a short abstract for the research paper titled: {title} that highlights the main ideas and essential results or conclusions?",
        f"Considering the title: {title}, could you author a summary that covers the main focus and underscores any noteworthy outcomes or conclusions?",
        f"Create a brief abstract for the research paper with the title: {title}, concentrating on the primary theme and any significant findings or conclusions?",
        f"Please create an abstract for a research paper using the title: {title}?",
        f"Could you produce a research paper abstract based on the following title: {title}?",
        f"Kindly generate a research paper abstract considering the title: {title}?",
        f"Compose an abstract for a research paper with the title: {title}?",
        f"Can you develop an abstract for a research paper derived from the title: {title}?",
        f"Please write an abstract for a research paper inspired by this title: {title}?",
        f"Craft a research paper abstract using the title: {title} as a basis?",
        f"Construct a research paper abstract corresponding to the title: {title}?",
        f"Create an abstract for a research paper based on the given title: {title}?",
        f"Formulate a research paper abstract that corresponds to the title: {title}?",
    ]

    prompt = random.choice(prompts)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an academic researcher."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2004,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["?"],
    )
    abstract = response.choices[0].message.content.strip()
    result.append(abstract)
    print(i)
    i = i + 1

    
Currect_df['abstract_ai'] = result
        
Currect_df.to_csv('data_gbt.csv', index=False)


Currect_df['abstracy'] = Currect_df['abstract']


Currect_df.to_pickle("df_big.pkl")  



df_mini = Currect_df[['title', 'abstract', 'abstract_ai', 'update_date']]


# create new dataframe to hold the split rows
new_df = pd.DataFrame(columns=['title', 'abstract', 'update_date', 'ai_generated'])

# iterate over each row in the original dataframe
for index, row in df_mini.iterrows():
    # create two new rows from each row in the original dataframe
    row1 = {'title': row['title'], 'abstract': row['abstract'], 'update_date': row['update_date'], 'ai_generated': False}
    row2 = {'title': row['title'], 'abstract': row['abstract_ai'], 'update_date': row['update_date'], 'ai_generated': True}
    
    # add the new rows to the new dataframe
    new_df = new_df.append(row1, ignore_index=True)
    new_df = new_df.append(row2, ignore_index=True)

new_df.to_csv('data_seperated_gbt.csv', index=False)

##ÆNDRE NAVN!!!
new_df.to_pickle("data_seperated_gbt.pkl")  

    



