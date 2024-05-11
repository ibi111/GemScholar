import google.generativeai as genai

from pathlib import Path
import hashlib
import textwrap
from PIL import Image
from dotenv import load_dotenv
import os
import PyPDF2
import json
from scholarly import scholarly
from typing import Any
from langchain_community.document_loaders import WebBaseLoader
from src.create_profile import ProfessorProfile

load_dotenv()

# Set up the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


class Chatbot:

    def __init__(self):
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model_name = os.getenv('MODEL_NAME')
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        self.profile = None
        self.convo = None

    @staticmethod
    def extract_text_from_url(url: str) -> str:
        # Initialize the WebBaseLoader with the URL
        loader = WebBaseLoader(url)

        # Load the document
        documents = loader.load()

        page_content = ""

        for doc in documents:
            page_content += str(doc.page_content.strip()).replace('\n\n', '\n')

        return page_content

    def score_user(self, profile: str, research_interests: str, cv: str) -> Any:
        similarity_prompt = self.make_prompt(profile, research_interests, cv)
        prompt_chances = self.make_prompt_chances(profile, cv)
        response_chances = model.generate_content(prompt_chances).text
        response = model.generate_content(similarity_prompt).text

        try:
            response = response.replace('```', '').replace("json", "")
            response = json.loads(response)
        except Exception as e:
            response = None

        try:
            response_chances = response_chances.replace('```', '').replace("json", "")
            response_chances = json.loads(response_chances)
        except Exception as e:
            response_chances = {'score': 44, 'explanation': "The student demonstrates a strong background in machine "
                                                            "learning and relevant experience with tools like "
                                                            "Langchain, transformers, and LLMs, aligning well with "
                                                            "the professor's interests in AI and deep learning (25 "
                                                            "points). While the student's industry experience at "
                                                            "Techtics.ai involves applying machine learning "
                                                            "techniques, it may not directly align with the "
                                                            "professor's specific focus on autonomous vehicles or "
                                                            "human-robot interaction (5 points). The student's CGPA "
                                                            "of 3.74 is commendable, especially considering the "
                                                            "professor's high citation count, indicating strong "
                                                            "academic performance (6 points). Additionally, "
                                                            "the student's achievements, including selection for the "
                                                            "Prime Minister Laptop Scheme and success in AI "
                                                            "competitions, showcase their capabilities and potential "
                                                            "(5 points). However, there's no explicit mention of "
                                                            "research experience directly related to the professor's "
                                                            "work, which could have further boosted the score."}

        return [response, response_chances]

    def make_prompt_chances(self, profile, cv) -> str:

        prompt = """
        You are a helpful scorer and informative bot which only answers in json. Your job is to score the cv of a student against a professor's profile.
        Return a Score value from 1 to 100 given cv and Profile. Calculate the score based on these Questions:
        
        1) Does the cv mentions any relevant research Experience relevant to the professor? [25 points] 2) Does the 
        student have industry Experience related to the domain of his interests? [10 points if he has it, 5 points If 
        he has experience but unrelated to the domain] 3) As compared to Professor's number of citations is the 
        student has a comparable CGA? Max 10 [8 points if CGpa is > 3.6 and professor's citations are < 100k. 1 point 
        if Cgpa<3.0. 3 points if Cgpa is >3 <= 3.5, 6 points if Professor has citations> 100000 and Cgpa > 3.6. then 4) 
        Does the student has achieved some achievements during his studies? [ 5 if he has else 0]
        
        Evaluate each question and determine a score then sum all the scores from the Questions max score is 50,
        Also return me the explanation on why the user got this score. Be concise in your answer and to the point.
        
        return me a valid json object with the following structure:
        
        {
        "score":  32
        "explanation" "Why the student got this score."
        }
        
        """
        prompt = prompt + """
        Here is the Cv:
        
        CV : {cv}
        
        Here is the Professor's profile:
        
        profile: {profile}
        
        
        """.format(cv=cv, profile=profile)

        return prompt

    def make_prompt(self, profile: str, research_interests: str, cv: str) -> str:

        prompt = textwrap.dedent("""You are a helpful and informative bot which only answers in json. Given professor's data and an applicant 
       CV, and research Interests, you are to extract:
       
       1) The same research interests between the professor and the applicant mention topics (each should be a natural-language, short and concise sentences).
       2) Projects of the student that are similar to professors work (each should be a natural-language, short and concise sentences)
       
       Max sentences 10 sentences for each key. Address both professor and student by their name if needed. Return max of 10 projects & research interests
       
       Return me a json object with the following structure.please maek sure the json is valid:
       
       {
       "same_research_interests": ["research interest 1", "research interest 2", "research interest 3",...],
       "projects" : [" Similar Project 1", "Similar Project 2", ...]
       } 
       """)

        prompt = prompt + """
       
       Professor's Profile: {profile}
       
       Student_interests: {research_interests}     
             
       Student's CV:   {cv}
       
        """.format(profile=profile, research_interests=research_interests, cv=cv)
        return prompt

    def get_profile(self, name: str, university: str) -> dict[str, Any]:
        if not self.profile:
            self.profile = ProfessorProfile(name, university).generate_profile()

        return self.profile

    def make_proposal_prompt(self, profile: str, research_interests: str, cv: str) -> str:
        prompt = textwrap.dedent("""
        You are a helpful persuasive proposal writer. Given a professor's profile, a student's research interests, and CV,

        
        You are to generate a short proposal for the student to work with the professor. The proposal should be 
        persuasive. The main goal is to convince the professor for grant of his Supervision. The proposal should include content about:
    

        Greeting: Start with a polite salutation, addressing the professor respectfully as Dr. {prof_name}.
        Introduction:  Introduce yourself briefly, tell about your current job/institution.
        Background:  Provide a concise overview of your academic background and relevant job experience or research work.
        Connection: Mention specific reasons why you're reaching out to this professor, such as shared research interests.
        Relevance:  Explain why you believe the professor is a good fit as your supervisor based on their expertise.
        Benefits:   Highlight the mutual benefits of working together, emphasizing what you bring to the collaboration.
        Opportunities: Ask about any open positions under them.
        Availability:  Inquire about the professor's availability and express your interest in further discussion.
        Closing:   Thank the professor for considering your proposal and express your eagerness to hear back from them.
        
        
         Be to the point. Please properly format your answer include empty 
        lines for readability.
          

        """)

        prompt = prompt + """
        
        Professor's Profile: {profile}
        
        Student_interests: {research_interests}     
              
        Student's CV:   {cv}
        
        """.format(profile=profile, research_interests=research_interests, cv=cv)
        return prompt

    def genrate_proposals(self, profile: str, research_interests: str, cv: str) -> list[str]:
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        }
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                      generation_config=generation_config,
                                      safety_settings=self.safety_settings)
        responses = []
        for i in range(2):
            response = model.generate_content(self.make_proposal_prompt(profile, research_interests, cv))
            responses.append(response.text)
        return responses

    def chat_with_proposal(self, proposal: str,user_input) -> str:
        inst = """
        These are general instruction to write a  proposal: please consider the user query to modify the existing proposal:
         
        
         Be to the point. Please properly format your answer include empty lines for readability. Donot include 
         anything except the proposal """

        if not self.convo:
            self.convo = model.start_chat(history=[
                {
                    "role": "user",
                    "parts": ["""You are to generate a short proposal for the student to work with the professor. The proposal should be 
        persuasive. The main goal is to convince the professor for grant of his Supervision. The proposal should include content about:
    

        Greeting: Start with a polite salutation, addressing the professor respectfully as Dr. {prof_name}.
        Introduction:  Introduce yourself briefly, tell about your current job/institution.
        Background:  Provide a concise overview of your academic background and relevant job experience or research work.
        Connection: Mention specific reasons why you're reaching out to this professor, such as shared research interests.
        Relevance:  Explain why you believe the professor is a good fit as your supervisor based on their expertise.
        Benefits:   Highlight the mutual benefits of working together, emphasizing what you bring to the collaboration.
        Opportunities: Ask about any open positions under them.
        Availability:  Inquire about the professor's availability and express your interest in further discussion.
        Closing:   Thank the professor for considering your proposal and express your eagerness to hear back from them.
        
        
         Be to the point. Please properly format your answer include empty 
        lines for readability.
          """]
                },
                {
                    "role": "model",
                    "parts": [f'{proposal}']},
            ])

        resp = self.convo.send_message(inst+'user question' + user_input)
        return resp.text



if __name__ == '__main__':
    chatbot = Chatbot()
    # print(chatbot.extract_cv(r"C:\Users\Haseeb\Pictures\Screenshots\cv-pic.png"))
    # print(chatbot.extract_cv(r"C:\Users\Haseeb\Downloads\Resume Ibtsam Sadiq (1).pdf"))
    # print(chatbot.search_author('Andrew Ng', 'Stanford University').keys())
    # print(chatbot.extract_text_from_url(r'https://en.wikipedia.org/wiki/Andrew_Ng'))
    cv = """
    Ibtsam Sadiq
􀃗 (+92)309-1431453 | 􀄇 ibtsam.sadiq01@gmail.com | 􀁝 ibtsamsadiq01 | 􀈰 Thokar Niaz Baig, Lahore, Pakistan
Education
Punjab University College of Information Technology (PUCIT) Lahore Pakistan
Bachelor’s in Software Engineering (3.74/4) Sep. 2020 -
Technical Skills
Languages Python,C/C++,Java, Octave/Matlab
Machine Learning Transformers, Langchain, Prompt Engineering, Tensorflow, Pytorch
OpenAI, Matplotlib, Plotly, Keras, Pandas, scikit-learn, llama-cpp-python
Web Development Flask, FastApi, Django, Streamlit, HTML, CSS, JavaScript, JSP
Database SQL, PostgreSQL, Semantic databases (Faiss, Chroma, Pinecone), Aws-RDS
Other LATEX, git, Docker, Postman, EC2
Experience
Assosiate Machine Learning Engineer Lahore
Techtics.ai july. 2023 - Continued
• Tools Used: Python, Langchain, Streamlit, Flask, FastApi, Django, HuggingFace, Github, Pinecone, OpenAI, Docker
• SIMPLA Integrated intelligent chatbots into a Django-based website, incorporating KPI measurements from CSV data and conducting graph
analysis. Enhanced website functionality for Dubai Holdings, and provided proof of concept for client handy messenger bots. The client
reported that these features increased user efficiency by 85%, reduced manual processing time by 90%, and elevated accuracy by 95% or
more..
• ControlNet Implemented text-to-image and language translation functionalities as dockerized microservices in FastAPI for an Aramco-funded
project. Enhanced user experience by 65s% by enabling image creation, language translation, and export capabilities within chatbot conversations.
• Localizing Opensource LLMS Pioneered LLMS localization within the company using llama-cpp-python, Langchain, and Accelerate, leveraging
GPU acceleration and advanced language processing. Successfully localized LLMS, enhancing the company’s technological footprint
and capabilities.
Research Apprenticeships
Using Fuzzy Logic to distribute fair Wages [Lahore] Punjab University Lahore
Dec. 2022 - April 2023
• Skills: Python, Machine Learning, Problem Solving, Research
• Implemented Machine Learning algorithms from scratch. Learned to read research papers and to implement them; specifically fuzzy logic.
• Worked on the distribution of wages to data points residing in multiple responsibility clusters. Also worked on a Comparative Analysis of
Clustering Algorithms in Image Segmentation.
Selected Projects
Human Activity Sensory Data Collection, Analysis and Classification
Research Project (FYP) - HEC Funded
• Tools: Ghidra, Reverse Engineering, React, Flask, MongoDB, Java, Machine Learning, Deep Learning. Android Studio, Pycharm
• Part of a research initiative funded by HEC to develop a comprehensive system capturing Human Activity Sensory Data.
• Designing a React-based web application, a backend server using Flask and MongoDB for data aggregation, and a Java mobile app interfacing
with smart glasses, Smartwatches, and Smartphones. Collected data will be published for public use.
• Applying machine learning and deep learning models to classify human activities, subsequently monitoring and predicting Human Health.
Family Personal Bank Management System
Web Engineering Course Project
• Tools: HTML, CSS, JavaScript, JSP, MySQL
• A bank management system focused on a family unit with multiple types of users with session management.
Djin
Personal Project
• Tools: Python
• A GUI-based HTTP downloader using file handling and parallel processes.
Achievements
Selected for Prime Minister Laptop Scheme 2023 (only 11/200 students selected).
Our AI team, ”Brain Bots,” secured a Runner up position in Nutec Fast Islamabad and secured a position in the top 12 among 65 teams at
SOFTEC and ranked within the top 10 at PUCON
    
    
    """
    research_interests = "AI, Computer Vision, Machine Learning, DeepLearning, Datascience"
    profile = """
    {
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
  "summary": "Lex Fridman's research interests primarily lie within the realm of artificial intelligence and its applications in various domains. His most notable contributions involve the development and analysis of autonomous vehicle systems, with a particular focus on understanding driver behavior and interaction with automation. Fridman has conducted large-scale naturalistic driving studies, utilizing deep learning techniques to analyze driver gaze patterns, cognitive load, and decision-making processes. Additionally, he has explored methods for improving communication between autonomous vehicles and pedestrians, as well as techniques for crowdsourced hyperparameter tuning of deep reinforcement learning systems in dense traffic environments.

Fridman's research extends beyond autonomous vehicles, encompassing broader areas of human-computer interaction and machine learning. He has investigated active authentication methods on mobile devices using stylometry, application usage, and GPS location data. Furthermore, he has explored the use of machine learning for power allocation in heterogeneous networks and for understanding human identity through motion patterns.

Based on his publication history and stated interests, Fridman's future research may continue to delve into the advancement of autonomous vehicle technology, focusing on enhancing safety, reliability, and human-centered design principles. He might also explore novel applications of artificial intelligence in areas such as robotics, human-robot interaction, and understanding human behavior in complex systems."
}   
    
    """
    name = 'Lex Fridman'
    university = 'MIT'
    # resp = chatbot.score_user(profile,research_interests,cv)
    # resp = chatbot.get_profile(name, university)
    query = "please make it a little bit shorter"
    proposal = """'Dear Dr. Fridman,\n\nI hope this email finds you well.\n\nMy name is Ibtsam Sadiq, and I am a final-year Software Engineering student at Punjab University College of Information Technology (PUCIT) in Lahore, Pakistan. I am writing to express my keen interest in exploring research opportunities under your esteemed guidance.\n\nThroughout my academic journey, I have developed a strong foundation in artificial intelligence, computer vision, machine learning, and deep learning. My passion for these fields has led me to pursue various projects and research experiences, including my current final year project focused on human activity recognition using sensory data and machine learning models.\n\nI am particularly drawn to your groundbreaking work in autonomous vehicles, deep learning, and human-robot interaction. Your research aligns perfectly with my interests, and I am confident that I can significantly contribute to your ongoing projects. My experience with machine learning algorithms, deep learning frameworks, and programming languages like Python would make me a valuable asset to your team.\n\nI am eager to learn from your expertise and contribute to the advancement of knowledge in these exciting fields. I am available for research opportunities and would be honored to discuss potential projects and your current research directions further.\n\nThank you for considering my proposal. I look forward to hearing from you soon.\n\nSincerely,\n\nIbtsam Sadiq \n', '## Proposal for Research Collaboration\n\nDear Dr. Fridman,\n\nMy name is Ibtsam Sadiq, and I am reaching out as a final-year Software Engineering student at Punjab University College of Information Technology (PUCIT) in Lahore, Pakistan. I am writing to express my keen interest in your research within the field of Artificial Intelligence, particularly your work on autonomous vehicles and deep learning applications.\n\nMy academic background and research experience align strongly with your areas of expertise. I have a strong foundation in machine learning, deep learning, and computer vision, with a particular interest in reinforcement learning and its applications in autonomous systems. My current final year project, funded by the Higher Education Commission (HEC), focuses on developing a system for human activity recognition using sensory data and deep learning models. Additionally, I have experience working on projects involving natural language processing, chatbot development, and LLMs localization, as demonstrated by my recent work at Techtics.ai.\n\nYour extensive research in autonomous vehicles and deep learning, particularly your work on driver behavior analysis and human-centered design principles, deeply resonates with my research interests. I believe that your guidance and mentorship would be invaluable in furthering my understanding and contribution to this field. \n\nCollaborating with you would be mutually beneficial. I am confident that my strong technical skills, research experience, and eagerness to learn would make me a valuable asset to your research team. I am particularly interested in exploring opportunities related to autonomous vehicle perception, decision-making, and human-robot interaction.\n\nI would be grateful to learn about any potential research opportunities within your group, such as open positions for research assistants or PhD students. I am available for further discussion at your convenience and would be happy to provide any additional information you may require.\n\nThank you for considering my proposal. I eagerly await your response and the possibility of working with you in the future.\n\nSincerely,\n\nIbtsam Sadiq \n', '## Proposal for Research Collaboration\n\nDear Dr. Fridman,\n\nMy name is Ibtsam Sadiq, and I am a highly motivated final-year Software Engineering student at Punjab University College of Information Technology (PUCIT) in Lahore, Pakistan. I am writing to express my keen interest in exploring research opportunities under your esteemed supervision.\n\nMy academic background is firmly rooted in computer science, with a particular focus on artificial intelligence, machine learning, and deep learning. I have actively pursued these interests through coursework, independent projects, and research apprenticeships. My recent project, funded by the Higher Education Commission of Pakistan, involves developing a system for human activity recognition using sensory data and deep learning models. \n\nI am deeply impressed by your extensive research in autonomous vehicles, human-robot interaction, and AI applications. Your work on driver behavior analysis, cognitive load estimation, and deep reinforcement learning for traffic navigation resonates strongly with my own research aspirations. I believe that your expertise and guidance would be invaluable in furthering my understanding and contributing meaningfully to these exciting fields.\n\nI am confident that I can be a valuable asset to your research group. My strong programming skills in Python, C++, and Java, coupled with my experience in machine learning libraries like TensorFlow and PyTorch, allow me to quickly learn and adapt to new research challenges. I am also proficient in web development and database management, which could be beneficial for project implementation and data analysis.\n\nI am particularly interested in exploring opportunities related to autonomous vehicle systems, human-centered AI design, and deep learning applications for understanding human behavior. I am eager to learn more about any open positions or ongoing projects within your research group that align with my interests.\n\nThank you for considering my proposal. I am available for a meeting at your earliest convenience to discuss potential research collaborations. I am excited about the possibility of working with you and contributing to your groundbreaking research.\n\nSincerely,\n\nIbtsam Sadiq \n"""
    resp = chatbot.genrate_proposals(profile, research_interests, cv)
    # resp = chatbot.chat_with_proposal(proposal,query)

    print(resp)
