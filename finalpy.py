import requests
import os
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st
import ollama
import chromadb
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")
       
model_l=load_embedding_model()

st.write("App started")
st.title("you tube learning assistant")
if "videos_data" not in st.session_state:
    st.session_state["videos_data"]=[]

if "summaries" not in st.session_state:
    st.session_state["summaries"]={}

if "Quizes" not in st.session_state:
    st.session_state["Quizes"]={}


if "questions" not in st.session_state: 
    st.session_state["questions"]={}


topic=st.text_input("Enter topic:")
search_clicked= st.button("search")
st.write("search button value =", search_clicked)

if search_clicked:   

    api_key=st.secrets["YOUTUBE_API_KEY"]

    url=f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={topic}&type=video&maxResults=4&key={api_key}"
    response=requests.get(url)

    search_data=response.json()

    video_ids=[]
    for item in search_data["items"]:
        video_id=item["id"].get("videoId")
        if video_id :
            video_ids.append(video_id)
    video_ids_str=",".join(video_ids)

    url_2=f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={video_ids_str}&key={api_key}"

    new_response=requests.get(url_2)
    stats_data=new_response.json()       
    stats_dict={}
    for item in stats_data["items"]:
        vid=item["id"]
        stats_one=item["statistics"]
        stats_dict[vid]={
        "views":int(stats_one.get("viewCount",0)),
        "likes":int(stats_one.get("likeCount",0)),
        "comments":int(stats_one.get("commentCount",0)),

        }
    # print(stats_dict)
    model=SentenceTransformer('paraphrase-multilingual-MiniLM-L12-V2') 
    st.write("step4")
            #solve a rank problem
    videos_data=[]
    for item in search_data["items"]:
    
       
        vid=item["id"]["videoId"]
        ttle=item["snippet"]["title"]
        ttle_similarity=cosine_similarity(model.encode([topic]),
                                        model.encode([ttle]))
        channel_name=item["snippet"]["channelTitle"]
        channel_bonus=0
        for word in topic.lower().split():
            if word in channel_name.lower():
                channel_bonus+=1
                break
        keyword_bonus=0
        title_lower=ttle.lower()
        for kw in topic.lower().split():
            if kw in title_lower:
                keyword_bonus+=1

        
        stats=stats_dict.get(vid,{})
        views=stats.get("views",0)
        likes=stats.get("likes",0)
        comments=stats.get("comments",0)
        videos_data.append(
            {"video_id":vid,
            "title":ttle,
            "views":views,
            "likes":likes,
            "comments":comments,
            "title_similarity":ttle_similarity,
            "channel_bonus":channel_bonus,
            "keyword":keyword_bonus,
            "channel":channel_name})
        
        
    if len(videos_data)==0:
        st.error("no videos added") 
        st.stop() 


    max_views=max(v["views"] for v in videos_data)
    max_likes=max(v["likes"] for v in videos_data)
    max_comments=max(v["comments"] for v in videos_data)

    if max_views !=0:
        views_norm=views/max_views
    else:
        views_norm=0

    if max_likes !=0:
        likes_norm= likes/max_likes
    else:
        likes_norm=0 

    if max_comments !=0:
        comments_norm=comments/max_comments
    else:
        comments_norm=0                


    if views!=0:
        relevance=likes/views
    else:
        relevance=0


    for v in videos_data:
        views_norm=v["views"]/max_views if max_views else 0
        likes_norm=v["likes"]/max_likes if max_likes else 0
        comments_norm=v["comments"]/max_comments if max_comments else 0
        relevance=v["likes"]/v["views"] if v["views"] else 0
        score=(
        0.10 * views_norm +
        0.05 * likes_norm +
        0.05  * comments_norm +
        0.10 * relevance+
        0.35 * v["title_similarity"][0][0]+
        0.20 * v["channel_bonus"]+
        0.15 * v["keyword"])

        v["score"]=round(score,2) 
    
    videos_data.sort(key =lambda x: x["score"] ,reverse=True)
   
    st.session_state["videos_data"]=videos_data
    
videos_data=st.session_state["videos_data"]

for i,video in enumerate (videos_data,start=1):
    thumbnail=f"https://img.youtube.com/vi/{video['video_id']}/0.jpg"

    col1,col2=st.columns([2,3])
    with col1:
        st.image(thumbnail,use_container_width=True)

    # st.write(thumbnail)
    with col2:
        st.subheader(f"{i}. {video['title']}")
        st.write(f"⭐ score : {video['score']:.2f}")
        st.write(f"📺 channel : {video['channel']}")

        st.write(f"👀 views:{video['views']:,}")
        st.write(f"👍 likes:{video['likes']:,}")
        st.write(f"💬 comments:{video['comments']:,}")

        with st.expander("Why recommended"):
            st.write(f"Title similarity: {video['title_similarity'][0][0]:.2f}")
            st.write(f"Keyword Match: {video['keyword']}")
            st.write(f"Channel Bonus: {video['channel_bonus']}")
            
        video_link=f"https://www.youtube.com/watch?v={video['video_id']}"

        st.link_button("📹 [watch video]", video_link)
        

        if st.button(f" summary ",key=f"summary_{i}"):
            st.write("button clicked")    
            api=YouTubeTranscriptApi()
            transcript=api.fetch(video["video_id"],languages=['hi','en','te'])
            st.success("Transcript loaded")
            results=[]
            full_text=" ".join([line.text for line in transcript]) 
            response=ollama.chat(model="llama3.2",
                                messages=[{"role":"user",
                                           "content" : f"""
                                                    Create:

                                                    Create output in this format:

                                                   SUMMARY:
                                                    A short summary in 100-200 words.

                                                    KEY POINTS:
                                                     bullet points(as per length of video).

                                                    IMPORTANT TERMS:
                                                    Important technical terms with one-line meanings.

                                                    INTERVIEW TAKEAWAYS:
                                                    3 important points for interviews or exams.


                                                    Transcript:
                                                    {full_text}
                                                    """ }])
    #st.write("Text created") 
            summary=response.message.content
            st.session_state["summaries"][video["video_id"]]=summary 
        if video["video_id"] in st.session_state["summaries"]:
            with st.expander("📄 Summary"):
                st.markdown(
                        st.session_state["summaries"][video["video_id"]]
                                                                        )
           
                        
        if st.button(f"📃 Quiz ",
                    key=f"quiz_{i}"):
            
            api=YouTubeTranscriptApi()
            transcript=api.fetch(video["video_id"],languages=['hi','en','te'])
            results=[]
            full_text=" ".join([line.text for line in transcript]) 
            response=ollama.chat(model="llama3.2",
                            messages=[{"role":"user",
                                        "content" : f""" 7
                                        . 
                                                Create:

                                                Create 5 multiple choice questions from this transcript.
                                                You need to take nextline after each option like i write in format.

                                                                Format:

                                                                Q1.
                                                                A)  (next line)
                                                                B)  (next line)
                                                                C)  (next line)
                                                                D)  (next line)
                                                                Answer:

                                                                Q2.
                                                                ...


                                                Transcript:
                                                {full_text}
                                                """ }])
            quiz=response.message.content
            st.session_state["Quizes"][video["video_id"]] = quiz
            

        if video["video_id"] in st.session_state["Quizes"]:
            with st.expander("📝 Quiz"):
                st.markdown(
                        st.session_state["Quizes"][video["video_id"]]
                                    )
          
        
        
        with st.expander("❓ Ask about this video "):
            question=st.text_input("🙋 Ask from this video",key=f"question_{i}")  
            if st.button(
               "🤖 Ask AI",
                    key=f"ask_{i}"
                                   ):
                st.success("Ask AI button clicked")
                api=YouTubeTranscriptApi()
                transcript=api.fetch(video["video_id"],languages=['hi','en','te'])
                

                chunks = []

                chunk_size = 600
                overlap_size = 120

                current_lines = []
                current_length = 0

                for line in transcript:

                    text = line.text.strip()

                    if not text:
                        continue

                    current_lines.append(line)
                    current_length += len(text)

                    if current_length >= chunk_size:

                        chunk_text = " ".join(
                            line.text.strip() for line in current_lines
                        )

                        chunks.append({
                            "text": chunk_text,
                            "start": current_lines[0].start,
                            "end": current_lines[-1].start + current_lines[-1].duration
                        })

                        overlap_lines = []
                        overlap_length = 0

                        for previous_line in reversed(current_lines):

                            overlap_lines.insert(0, previous_line)

                            overlap_length += len(previous_line.text)

                            if overlap_length >= overlap_size:
                                break

                        current_lines = overlap_lines
                        current_length = overlap_length


                


                
               
                
              
                embeddings=model_l.encode([chunk["text"] for chunk in chunks],normalize_embeddings=True)
                
                # CHROMADB IMPLEMENTATION
                client=chromadb.PersistentClient(path="./chroma_db")
                collection=client.get_or_create_collection( name="YOU-TUBE",
                                                    metadata={"hnsw:space":"cosine"})
            
                collection.upsert(
                    documents=[chunk["text"] for chunk in chunks],
                    embeddings=embeddings.tolist(),
                    metadatas =[
                            {
                                "video_id": video["video_id"],
                                "start": chunk["start"],
                                "end": chunk["end"]
                            }
                            for chunk in chunks
                        ],

                    ids=[f"{video['video_id']}_{i}"
                          for i in range(len(chunks))]         ) 

                question_embedding=model_l.encode(question,
                                                  normalize_embeddings=True)


                # match the best chunk
                result=collection.query(
                   query_embeddings=[question_embedding.tolist()],
                   n_results=5,
                   where ={"video_id":video['video_id']})

            

                documents = result["documents"][0]
                distances = result["distances"][0]
                metadatas = result["metadatas"][0]

                threshold = 0.9

                relevant_chunks = []
                relevant_metadata = []

                for doc, dist, meta in zip(documents, distances, metadatas):

                    if dist < threshold:
                        relevant_chunks.append(doc)
                        relevant_metadata.append(meta)


                if not relevant_chunks:

                    st.warning("This topic is not discussed in video")

                else:

                    st.write("### 📌 Relevant sections")

                    for doc, metadata in zip(relevant_chunks, relevant_metadata):

                        start = metadata["start"]
                        end = metadata["end"]

                        video_id = metadata["video_id"]

                        start_seconds = int(start)

                        youtube_url = (
                            f"https://www.youtube.com/watch?v={video_id}"
                            f"&t={start_seconds}s"
                        )

                        minutes = start_seconds // 60
                        seconds = start_seconds % 60

                        end_seconds = int(end)
                        end_minutes = end_seconds // 60
                        end_seconds = end_seconds % 60

                        st.markdown(
                            f"▶️ [{minutes}:{seconds:02d} - "
                            f"{end_minutes}:{end_seconds:02d}]({youtube_url})"
                        )

                        st.write(doc)
                        st.divider()


                context = "\n\n".join(relevant_chunks)

                prompt = f"""
                                Answer ONLY from the context.

                                Context:

                                {context}

                                Question:

                                {question}

                                """
                response=ollama.chat(
                        model="llama3.2",
                        messages=[{

                             "role":"user",
                            "content":prompt


                       } ]
              
                        

                )
                answer=response.message.content
                st.markdown(answer)
        st.divider()

    

