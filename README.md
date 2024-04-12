# Lyrics-Generator
Lyrics generator using LLMs

<b> <u> INTRODUCTION</u> :</b>

  - Lyrics are an essential component of music. Musicians can communicate their ideas, feelings, and experiences to their listeners through lyrics. But creating lyrics can be difficult since it requires a skill set combining creativity, language mastery, and cultural sensitivity. 
  - The recent development in Natural Language Processing over the years has made it possible to generate sentences based on specific prompts. NLP models may analyze and synthesize language in a way that imitates human creativity by utilizing the capabilities of machine learning and computational linguistics.

> OBJECTIVE:
    - The goal of this project is to create a high-caliber, entertaining lyrics generator that can work with a variety of musical genres. 
    - We also want to investigate the many types of lyrics based on genre-specific famous artists through this project.

* Choosen Genres: 
  * Country 
  * Pop 
  * Rhythms and Blues (RB) 
  * Rock 
  * Rap 

  We also chose 5 different artists for each particular genre to fine-tune artist-specific models. 
________________________________________________

<b> <u>PROJECT FLOW</u> : </b>

PART A : Exploratory Data Analysis 
> 
  <b> 1) Data Collection </b>
  #### We used the Kaggle - "English songs lyrics" dataset for our project (https://www.kaggle.com/datasets/razauhaq/english-songs-lyrics?select=df_eng.csv). This dataset is the preprocessed version of 5 Million Song Lyrics   Dataset contaning lyrics of only English songs extracted by CARLOSGDCJ.
  
  ##### The dataset characteristic is as:
   
  
  * It consisted of 3373529 rows and 6 columns.
  * The six columns are namely: 'title', 'tag', 'artist', 'year', 'views', 'lyrics'
  * The 'tag' denotes the genre of the lyrics - pop, rap, rock, rb, misc, country 
  
  We used Kaggle notebook to divide our dataset into sub_dataset consisting of individual genres. Below are the code snippets used in Kaggle:
  
  
  ```bash
  df_eng = pd.read_csv('/kaggle/input/english-songs-lyrics/df_eng.csv', index_col = False)
  
  # Segregate data into specific genres:
  part_pop = df_eng.loc[df_eng['tag'] == 'pop']
  part_rock = df_eng.loc[df_eng['tag'] == 'rock']
  part_rap = df_eng.loc[df_eng['tag'] == 'rap']
  part_rb = df_eng.loc[df_eng['tag'] == 'rb']
  part_misc = df_eng.loc[df_eng['tag'] == 'misc']
  part_country = df_eng.loc[df_eng['tag'] == 'country']
  
  # Store the segregated data in pickle files
  part_pop.to_pickle('pop.pickle')
  part_rock.to_pickle('rock.pickle')
  part_rap.to_pickle('rap.pickle')
  part_rb.to_pickle('rb.pickle')
  part_misc.to_pickle('misc.pickle')
  part_country.to_pickle('country.pickle')
  ```
  
  **We used Kaggle for this initial seggregation due to inability of Colab to process a bigger dataset.
  
  <b> 2) Data Pre-processing </b>
    (i) We pre-processed each genre_dataset file by removing the inconsistent records from those datasets.
    Such as:
    
    
    *   Removing those records where 'year' of release of song is before the 1900s 
    *   Removing those records where number of songs by the artists is less than 5 or 10 in number.
    
    <br>*These steps were taken to reduce the dataset. As most of the genres consisted near about 10,00,000 datapoints and this was too big for the model to train on and also we had limited resources available with us.
    <br>**This might introduce bias to our models to certain extent as we have removed the minority representation from our dataset. But given the limitations of this project, we had to take certain steps to limit our dataset.*
  
  <b> 3) Data Visualization </b>

>
________________________________________________
PART B : Language Modeling
> 
  * Model Comparison and Evaluation (Bi-directional LSTM, GPT2, DistilGPT2) using ROUGE Metric and Human Evaluation.
  * Genre Specific Training
  * Artist Specific Fine-Tuning
  * Lyrics Generations
_____________________________________________
