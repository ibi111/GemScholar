import streamlit as st
import PIL.Image
import google.generativeai as genai
import os
import PyPDF2
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from src.chatbot import Chatbot
import json



load_dotenv()


data = {
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
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def extract_cv(uploaded_file) -> str:
    # Define the path where the file will be saved
    file_path = os.path.join(os.getcwd(), uploaded_file.name)

    # Write the uploaded file's content to the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.jpg'):
            # Open the image file using PIL
            obj = PIL.Image.open(file_path)
            response = st.session_state.model.generate_content([obj, "Give me all the text of the this image"])
            return response.text
        elif uploaded_file.name.endswith('.pdf'):
            # Process PDF files
            return extract_pdf_pages(file_path)
        else:
            raise NotImplementedError("File type not supported")
    finally:
        # Optionally remove the file if you don't want to keep it
        os.remove(file_path)  # Ensure the temporary file is deleted

def extract_pdf_pages(pathname: str) -> str:
    parts = [f"--- START OF PDF {pathname} ---"]
    with open(pathname, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            parts.append(f"--- PAGE {page_number + 1} ---")
            parts.append(page.extract_text())  # Extract text from the page
    return "\n".join(parts)

# The rest of your Streamlit app's code remains unchanged

def main():
    col1, col2 = st.columns([1, 5])  # Adjust the ratio based on your aesthetic preference

    # Assuming 'logo.png' is in the same directory as your script or provide the full path
    logo_path = 'logo.png'  # or URL to the logo image

    with col1:
        st.image(logo_path, width=100)  # You can adjust the width to fit your layout

    with col2:
        st.title("GemScholar")
    

    if 'bot' not in st.session_state:
        st.session_state.bot =  Chatbot()

    # Check if we are on the main page, the summary page, or the draft page
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    if 'model' not in st.session_state:
        st.session_state.model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

    if st.session_state.page == 'main':
        display_main_page()
    elif st.session_state.page == 'summary':
        display_summary_page()
    elif st.session_state.page == 'relevance':
        display_relevance_page()
    elif st.session_state.page == 'chat_with_proposal':
        display_chat_with_proposal_page()
    elif st.session_state.page == 'drafts':
        display_drafts_page()

def display_main_page():
    professor_name = st.text_input("Professor Name")
    university_name = st.text_input("University Name")
    profile_url = st.text_input("Professor Profile URL (Optional)")
    uploaded_file = st.file_uploader("Upload Your CV or Resume", type=['pdf', 'docx', 'png', 'jpg'])
    interests = st.text_area("Enter your interests", height=300)

    if st.button("Next"):
        if not professor_name or not university_name or not interests:
            st.error("Please fill in all required fields.")
        else:
            # Save inputs to session state
            st.session_state.professor_name = professor_name
            st.session_state.university_name = university_name
            st.session_state.profile_url = profile_url
            st.session_state.interests = interests
            st.session_state.uploaded_file = uploaded_file
            if st.session_state.uploaded_file is not None:
                with st.spinner('Extracting CV...'):
                    extracted_data = extract_cv(st.session_state.uploaded_file)
                    st.session_state.extracted_cv = extracted_data
            st.session_state.page = 'summary'
            st.experimental_rerun()

def display_summary_page():
    with st.spinner("Generating Summary..."):
        st.session_state.profile = st.session_state.bot.get_profile(st.session_state.professor_name, st.session_state.university_name)
        # st.session_state.profile = data
        print(st.session_state.profile) 
        # # Debugging output to check the type of profile_data
        # st.write("Debug - Profile Data Type:", type(profile_data))
        # st.write("Debug - Profile Data:", profile_data)

        # # Check if profile_data is already a dictionary
        # if isinstance(profile_data, dict):
        #     data = profile_data
        # elif isinstance(profile_data, str):
        #     try:
        #         data = json.loads(profile_data)
        #     except json.JSONDecodeError as e:
        #         st.error(f"Failed to parse profile data: {e}")
        #         return  # Exit the function if parsing fails
        # else:
        #     st.error("Unexpected data type for profile data.")
        #     return  # Exit the function if data type is not handled

        # st.session_state.profile = profile_data
        with st.sidebar:
            st.write("Submission Received")
            st.write(type(st.session_state.profile))
            st.write("Professor Name:", st.session_state.professor_name)
            st.write("University Name:", st.session_state.university_name)
            st.write("Profile URL:", st.session_state.profile_url)
            st.write("Interests:", st.session_state.interests)
            st.write("Extracted Data is ", st.session_state.extracted_cv)
            if st.session_state.uploaded_file is not None:
                st.write("File Uploaded:", st.session_state.uploaded_file.name)

        with st.sidebar:
            st.write("Submission Received")
            st.write("Professor Name:", st.session_state.profile['author_name'])
            st.write("University Name:", st.session_state.profile['university'])
            # st.write("Profile URL:", st.session_state.profile['homepage'])

        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.profile['url_picture'], caption=st.session_state.profile['author_name'])
        with col2:
            st.write(f"### {st.session_state.profile['author_name']}")
            st.write(f"**University:** {st.session_state.profile['university']}")
            st.write(f"**Email Domain:** {st.session_state.profile['email_domain']}")

        st.write("### H-Index")
        st.write(f"***{st.session_state.profile['h-index']}***")

        st.write("### Total Publications")
        st.write(f"***{st.session_state.profile['number_of_publications']}***")

        st.write("### Summary")
        st.write(st.session_state.profile['summary'])

        st.write("### Interests")
        for idx, interest in enumerate(st.session_state.profile['interests'], start=1):
            st.write(f"{idx}. {interest}")

        st.write("### Top 5 Publications")
        for publication in st.session_state.profile['latest_5_publications']:
            st.write(f"- {publication['title']} ({publication['year']}) - Citations: {publication['citations']}")

        

        # Buttons for navigation
        if st.button("Go Back"):
            st.session_state.page = 'main'
            st.experimental_rerun()
        if st.button("Check Your Relevamce"):
            st.session_state.page = 'relevance'
            st.experimental_rerun()


def display_relevance_page():
    # Simulated data received from the backend
    # relevance_data = [
    #     {
    #         'same_research_interests': [
    #             'Lex Fridman and the student share a common interest in Artificial Intelligence.',
    #             'Both Lex Fridman and the student are interested in Deep Learning techniques.',
    #             'They both have research interests in Machine Learning applications.'
    #         ],
    #         'projects': [
    #             "The student's project on 'Human Activity Sensory Data Collection, Analysis, and Classification' aligns with Lex Fridman's work on understanding human behavior and applying machine learning to analyze complex systems.",
    #             "The student's experience with 'Fuzzy Logic to distribute fair Wages' demonstrates their understanding of machine learning algorithms, which is relevant to Lex Fridman's research in artificial intelligence and its applications."
    #         ]
    #     },
    #     {
    #         'score': 43,
    #         'explanation': "The student demonstrates strong potential with relevant experience in machine learning and deep learning, aligning well with the professor's interests. The student's CGPA of 3.74 is impressive, especially considering the professor's high citation count. While the research experience isn't a perfect match, the project involving human activity recognition showcases relevant skills. The student's achievements and industry experience further strengthen their profile."
    #     }
    # ]
    professor_profile_json = st.session_state.profile  # JSON data
    user_interests = st.session_state.interests  # User's interests as string
    user_cv = st.session_state.extracted_cv  # Extracted CV as string

    # Convert JSON profile to string if it's not already
    professor_profile_str = json.dumps(professor_profile_json) if isinstance(professor_profile_json, dict) else professor_profile_json

    relevance_data = st.session_state.bot.score_user(professor_profile_str, user_interests, user_cv)

    with st.sidebar:
        st.write(relevance_data)

    # Extracting data for display
    interests_projects = relevance_data[0]
    acceptance_data = relevance_data[1]

    # Create columns for the pie chart and the explanation
    col1, col2 = st.columns(2)

    # Displaying the pie chart for the score in the first column
    with col1:
        fig, ax = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
        # Set the background color
        fig.patch.set_facecolor('#121212')  # Dark background for the figure
        ax.set_facecolor('#121212')  # Dark background for the axes

        ax.pie([acceptance_data['score'], 50 - acceptance_data['score']], labels=['Chances', 'Unlikely'], autopct='%1.1f%%',
               startangle=90, colors=['#1f77b4', '#ff7f0e'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    # Displaying the explanation in the second column
    with col2:

        st.subheader("Explanation for Score")
        st.write(acceptance_data['explanation'])

    st.subheader("Your Relevance in Research Interests")
    for idx, interest in enumerate(interests_projects['same_research_interests'], start=1):
        st.write(f"{idx}. {interest}")

    st.subheader("Alignment in Projects")
    for idx, project in enumerate(interests_projects['projects'], start=1):
        st.write(f"{idx}. {project}")

    # Buttons for navigation
    if st.button("Go Back"):
        st.session_state.page = 'summary'
        st.experimental_rerun()
    if st.button("Generate Mail Drafts"):
        st.session_state.page = 'drafts'
        st.experimental_rerun()

def display_drafts_page():
    st.subheader("Generated Email Drafts")
    if 'profile' in st.session_state and 'interests' in st.session_state:
        # proposals = st.session_state.bot.genrate_proposals(st.session_state.profile, st.session_state.interests, st.session_state.extracted_cv)
        proposals= ["Dear Dr. Fridman,\n\nI hope this email finds you well.\n\nMy name is Ibtsam Sadiq, and I am currently an Associate Machine Learning Engineer at Techtics.ai in Lahore. I am writing to express my keen interest in exploring potential research opportunities under your supervision at MIT.\n\nI recently completed my Bachelor's degree in Software Engineering from Punjab University College of Information Technology (PUCIT), where I maintained a strong academic record. My passion lies at the intersection of artificial intelligence, computer vision, and machine learning. My current role at Techtics.ai involves developing and implementing AI-driven solutions for various clients, including projects related to chatbot integration, text-to-image generation, and language translation.\n\nI am particularly impressed by your extensive work in autonomous vehicles, deep learning, and human-robot interaction. Your research aligns perfectly with my own interests, and I believe that your expertise would be invaluable in guiding my research aspirations. I am confident that my strong technical skills, research experience, and eagerness to learn would make me a valuable asset to your team. \n\nI am particularly interested in exploring opportunities related to:\n\n*   Deep learning applications in autonomous vehicles\n*   Human-centered design principles for AI systems\n*   Crowdsourced approaches to AI development\n\nI am available for research collaboration and would be happy to discuss potential projects further. Please let me know if there are any open positions or opportunities for collaboration within your research group.\n\nThank you for considering my proposal. I eagerly await your response and the opportunity to discuss how I can contribute to your groundbreaking research.\n\nSincerely,\n\nIbtsam Sadiq \n", '## Proposal to Dr. Lex Fridman for Research Supervision\n\nDear Dr. Fridman,\n\nMy name is Ibtsam Sadiq, and I am reaching out as a final-year Software Engineering student at Punjab University College of Information Technology (PUCIT) in Lahore, Pakistan, with a strong interest in pursuing research in Artificial Intelligence.\n\nThroughout my academic journey, I have cultivated a deep passion for AI, Computer Vision, Machine Learning, Deep Learning, and Data Science. This is evident in my academic projects, including my current final year project, "Human Activity Sensory Data Collection, Analysis and Classification," funded by the Higher Education Commission (HEC) of Pakistan. This project involves developing a system for capturing human activity data using smart devices and employing machine learning models for activity classification and health monitoring.\n\nI am particularly drawn to your research at MIT, especially your work on autonomous vehicles, deep learning, and human-computer interaction. Your research on driver behavior analysis, cognitive load estimation, and pedestrian communication aligns perfectly with my interests in understanding and developing intelligent systems that interact seamlessly with humans. \n\nI am confident that my skills in Python, Machine Learning libraries (TensorFlow, PyTorch, scikit-learn), and experience with projects like "SIMPLA" (intelligent chatbot integration) and "ControlNet" (text-to-image and language translation) would make me a valuable asset to your research group. I am eager to learn from your expertise and contribute to cutting-edge research in AI.\n\nI am writing to inquire about potential research opportunities within your group. I am available for full-time research starting in September 2020 and am flexible with project timelines. I would be grateful for the opportunity to discuss my research interests and potential contributions in more detail. \n\nThank you for considering my proposal. I eagerly await your response and the possibility of working under your esteemed guidance.\n\nSincerely,\n\nIbtsam Sadiq \n']
        cols = st.columns(2)  # Create two columns for side-by-side layout
        for i, draft in enumerate(proposals, start=1):
            with cols[i % 2]:  # Alternate between columns
                st.text_area(f"Draft {i}", value=draft, height=300, key=f"draft_{i}")
                if st.button(f"Discuss Draft {i}", key=f"btn_discuss_{i}"):
                    st.session_state.current_proposal = draft
                    st.session_state.page = 'chat_with_proposal'
                    st.experimental_rerun()
    else:
        st.error("Profile or interests data is missing.")

    if st.button("Go Back"):
        st.session_state.page = 'relevance'
        st.experimental_rerun()
def display_chat_with_proposal_page():
    st.subheader("Chat about the Proposal")
    if 'current_proposal' in st.session_state:
        st.write("Discussing the Proposal:")
        st.text_area("Proposal", value=st.session_state.current_proposal, height=300, disabled=True)
        if "messages" not in st.session_state.keys():  # Initialize the chat message history
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi 👋👋👋 , How can i help you today ?"}
            ]

        if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.bot.chat_with_proposal(st.session_state.current_proposal, prompt)
                    print(response)
                    message = {"role": "assistant", "content": response}
                    
                    st.session_state.messages.append(message)
                    st.experimental_rerun()

        
    else:
        st.error("No proposal is currently selected.")

    if st.button("Go Back to Drafts"):
        st.session_state.page = 'drafts'
        st.experimental_rerun()

def copy_to_clipboard(text):
    st.experimental_set_query_params(text=text)  # This is a placeholder for actual clipboard functionality
    st.info("Copied to clipboard!")

if __name__ == "__main__":
    main()