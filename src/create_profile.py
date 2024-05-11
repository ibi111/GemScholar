import os

import google.generativeai as genai
from src.ProfessorSearch import ProfessorSearch
import json

data =  {
  "author_name": "Lex Fridman",
  "university": "MIT",
  "affiliation": "MIT",
  "email_domain": "@mit.edu",
  "interests": [
    "Artificial Intelligence",
    "Deep Learning",
    "Autonomous Vehicles",
    "Human-Robot Interaction",
    "Reinforcement Learning"
  ],
  "number_of_publications": 44,
  "citedby": 2892,
  "h-index": 23,
  "i10-index": 37,
  "url_picture": "https://scholar.google.com/citations?view_op=medium_photo&user=wZH_N7cAAAAJ",
  "cites_per_year": {
    "2012": 14,
    "2013": 6,
    "2014": 17,
    "2015": 34,
    "2016": 77,
    "2017": 113,
    "2018": 204,
    "2019": 410,
    "2020": 510,
    "2021": 486,
    "2022": 457,
    "2023": 391,
    "2024": 125
  },
  "homepage": "https://lexfridman.com/",
  "top_10_publications": [
    {
      "title": "DeepTraffic: Crowdsourced Hyperparameter Tuning of Deep Reinforcement Learning Systems for Multi-Agent Dense Traffic Navigation",
      "year": 2018,
      "citations": 391
    },
    {
      "title": "MIT advanced vehicle technology study: Large-scale naturalistic driving study of driver behavior and interaction with automation",
      "year": 2019,
      "citations": 320
    },
    {
      "title": "Active authentication on mobile devices via stylometry, application usage, web browsing, and GPS location",
      "year": 2016,
      "citations": 245
    },
    {
      "title": "Learning human identity from motion patterns",
      "year": 2016,
      "citations": 211
    },
    {
      "title": "Driver gaze region estimation without use of eye movement",
      "year": 2016,
      "citations": 176
    },
    {
      "title": "To walk or not to walk: Crowdsourced assessment of external vehicle-to-pedestrian displays",
      "year": 2017,
      "citations": 157
    },
    {
      "title": "Cognitive load estimation in the wild",
      "year": 2018,
      "citations": 149
    },
    {
      "title": "A machine learning approach for power allocation in HetNets considering QoS",
      "year": 2018,
      "citations": 143
    },
    {
      "title": "Owl and Lizard: patterns of head pose and eye pose in driver gaze classification",
      "year": 2016,
      "citations": 106
    },
    {
      "title": "Human-centered autonomous vehicle systems: Principles of effective shared autonomy",
      "year": 2018,
      "citations": 88
    }
  ],
  "summary": """Lex Fridman's research interests primarily lie within the realm of artificial intelligence and its applications in various domains. His most notable contributions involve the development and analysis of autonomous vehicle systems, with a particular focus on understanding driver behavior and interaction with automation. Fridman has conducted large-scale naturalistic driving studies, utilizing deep learning techniques to analyze driver gaze patterns, cognitive load, and decision-making processes. Additionally, he has explored methods for improving communication between autonomous vehicles and pedestrians, as well as techniques for crowdsourced hyperparameter tuning of deep reinforcement learning systems in dense traffic environments.

Fridman's research extends beyond autonomous vehicles, encompassing broader areas of human-computer interaction and machine learning. He has investigated active authentication methods on mobile devices using stylometry, application usage, and GPS location data. Furthermore, he has explored the use of machine learning for power allocation in heterogeneous networks and for understanding human identity through motion patterns.

Based on his publication history and stated interests, Fridman's future research may continue to delve into the advancement of autonomous vehicle technology, focusing on enhancing safety, reliability, and human-centered design principles. He might also explore novel applications of artificial intelligence in areas such as robotics, human-robot interaction, and understanding human behavior in complex systems."""
}

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

class ProfessorProfile:
    def __init__(self, professor_name, university):
        self.professor_name = professor_name
        self.university = university
        self.professor_profile = None

    def generate_profile(self):

        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # searching author
        author_search = ProfessorSearch(self.professor_name, self.university)
        author_full = author_search.search_author()

        # stringify the author details
        author_details = str(author_full)

        # print(author_details)

        example_profile = """{
                    "author_name": "Robert Tibshirani",
                    "university": "Stanford University",
                    "affiliation"': 'Professor of Biomedical Data Sciences, and  of  Statistics, Stanford University
                    "email_domain": '@stanford.edu',
                    "interests": ['Statistics', 'data science', 'Machine Learning'],
                    "number_of_publications": 100,
                    "citedby": 490518,
                    "h-index": 100,
                    "i10-index": 100,
                    "url_picture"': 'https://scholar.google.com/citations?view_op=medium_photo&user=ZpG_cJwAAAAJ'
                    "cites_per_year": { 
                                        2019: 32781,
                                        2020: 35771,
                                        2021: 39823,
                                        2022: 39021,
                                        2023: 38623,
                                        2024: 12883
                                        },
                    "homepage": 'https://statweb.stanford.edu/~tibs/'

                    latest_5_publications": [
                        {
                            "title": "Using Pre-training and Interaction Modeling for ancestry-specific disease prediction in UK Biobank",
                            "year": 2024,
                            "citations": 0
                        },
                        {
                            "title": "CAR19 monitoring by peripheral blood immunophenotyping reveals histology-specific expansion and toxicity.,
                            "year": 2024,
                            "citations": 0
                        }
                    ],

                    "top_3_publications": [
                        {
                            "title": "Title of the publication",
                            "year": 2021,
                            "citations": 100
                        },
                        {
                            "title": "Title of the publication",
                            "year": 2021,
                            "citations": 100
                        }
                    ]
                    "summary": "Robert Tibshirani's current research focuses on developing statistical methods and machine learning algorithms for solving complex problems in various fields, including genomics, neuroscience, and personalized medicine. He is particularly interested in developing techniques for analyzing high-dimensional data, where the number of variables greatly exceeds the number of observations.
                                Tibshirani is best known for his work on regularization methods such as the LASSO and elastic net, which are widely used for feature selection and prediction in regression analysis. He has also contributed to the development of methods for clustering, classification, and survival analysis.
                                In terms of future research interests, Tibshirani may continue exploring novel statistical techniques for analyzing emerging types of data, such as single-cell sequencing data in genomics or functional MRI data in neuroscience. Additionally, he may be interested in developing methods for addressing challenges related to interpretability, scalability, and robustness in statistical learning algorithms."
                }"""

        # generating profile of author based on the author details fetched from the search
        prompt = f"""
                You are a helpful json profile maker. Based on information provided please make a profile of the author.

                Include the details of the author like the number of publications, h-index, i10-index, and the top 3.Also
                include the latest 5 publications of the author on the basis of 'pub_year', with the title, year of publication,
                and the number of citations. If no citation is available, mention it as 0.
                Also add the summary of regarding what he/she is currently researching on and what he/she is most researches 
                about. In what kind of topics of research he can be interested for working on.

                All the profile content must be from the author details fetched from the Google Scholar Profile. Don't
                include any information from yourself. Return output in valid json format.
                               
                Here are the details of {self.professor_name} at {self.university} fetched using scholarly 
                package from his/her Google Scholar Profile. 
                Author Details : {author_details}
                Based on the details, create a profile of {self.professor_name} at {self.university}.
                
                
                Please provide the output in the following format:
                Example of the profile: {example_profile}

                """

        response = model.generate_content(prompt).text

        try:
            response = response.replace('```', '').replace("json", "")
            response = json.loads(response)
        except Exception as e:
            response =  data


        self.professor_profile = response
        return self.professor_profile

        # response = model.generate_content(f"Create a profile of {self.professor_name} at {self.university}")
        # self.professor_profile = response.text
        # return self.professor_profile


if __name__ == '__main__':
    genai.configure(api_key="AIzaSyC47Nnkq9tFdc2rURov7qw8Odbx8DS267g")
    professor_name = 'Lex Fridman'
    university = 'MIT'
    professor_profile = ProfessorProfile(professor_name, university)
    profile = professor_profile.generate_profile()
    print(profile)