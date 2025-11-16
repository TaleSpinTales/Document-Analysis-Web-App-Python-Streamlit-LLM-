css = '''
<style>
body {
    background-color: #f4f4f4;
    font-family: Arial, sans-serif;
}
h1, .stTextInput, .stButton {
    text-align: center !important;
}
.stTextInput > div > div {
    margin: 0 auto;
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    justify-content: center;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://miro.medium.com/v2/resize:fit:933/0*IL4DfVs_zQspM7uq.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://wallpapers.com/images/featured/unknown-png-2fudx37oj7ug3d8l.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
