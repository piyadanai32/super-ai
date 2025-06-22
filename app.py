import os
import json
import logging
import re
from datetime import datetime
from flask import Flask, request, abort, jsonify
from dotenv import load_dotenv  
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    TextMessage, FlexMessage, QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.exceptions import InvalidSignatureError
from google.protobuf.json_format import MessageToDict

from retriever import search_from_documents
from dialogflow import detect_intent_texts
from message import (
    process_payload, create_flex_message,
    send_multiple_messages, send_text_message
)

# โหลด environment variables จากไฟล์ .env
load_dotenv()

# ตรวจสอบและกำหนดค่าตัวแปรสภาพแวดล้อมที่จำเป็น
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DIALOGFLOW_PROJECT_ID = os.getenv("DIALOGFLOW_PROJECT_ID")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# กำหนดค่า Config
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
SESSION_ID = "line-bot-session"

# Flask App
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = app.logger

# Line Bot
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)

# คำตอบที่ไม่ต้องการจาก Dialogflow (หากได้คำตอบเหล่านี้จะถือว่า Dialogflow ไม่สามารถตอบคำถามได้)
INVALID_DIALOGFLOW_RESPONSES = [
    "ขอโทษค่ะ พูดอีกครั้งได้ไหมคะ",
    "ขอโทษค่ะ ไม่เข้าใจ",
    "พูดใหม่อีกครั้งได้ไหมคะ",
    "ขอโทษค่ะ ฉันไม่เข้าใจค่ะ",
    "พูดอีกทีได้ไหมคะ" 
]

@app.route("/callback", methods=['POST'])
def callback():
    """
    เส้นทาง webhook สำหรับรับข้อความจาก LINE
    """
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    logger.info(f"ได้รับคำขอ: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("ลายเซ็นไม่ถูกต้อง")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    text_from_user = event.message.text
    logger.info(f"ข้อความจาก {user_id}: {text_from_user}")

    # ตรวจสอบและแยกข้อความอย่างมีประสิทธิภาพ
    is_group = hasattr(event.source, 'type') and event.source.type in ['group', 'room'] 
    bot_name = "DMC Chatbot"
    actual_message = text_from_user.split(f'@{bot_name}', 1)[-1].strip() if is_group and text_from_user.startswith(f'@{bot_name}') else text_from_user
    should_respond = not is_group or (is_group and text_from_user.startswith(f'@{bot_name}'))

    if not should_respond or not actual_message:
        if should_respond:
            send_text_message(line_bot_api, event.reply_token, f"สวัสดีค่ะ หนูชื่อ {bot_name} คุณต้องการสอบถามอะไรค่ะ?")
        return

    try:
        # ส่งคำถามไปยัง Dialogflow
        response = detect_intent_texts(DIALOGFLOW_PROJECT_ID, f"{SESSION_ID}-{user_id}", actual_message, 'th')
        response_dict = MessageToDict(response._pb)
        
        messages_to_reply = []
        quick_replies = None
        has_payload = False
        
        if 'queryResult' in response_dict:
            # ตรวจสอบและรวบรวมทุกข้อความจาก Dialogflow
            if 'fulfillmentMessages' in response_dict['queryResult']:
                for message in response_dict['queryResult']['fulfillmentMessages']:
                    # ตรวจสอบข้อความปกติ
                    if 'text' in message and 'text' in message['text'] and message['text']['text']:
                        for text in message['text']['text']:
                            if text and text not in INVALID_DIALOGFLOW_RESPONSES:
                                text_message = TextMessage(text=text)
                                messages_to_reply.append(text_message)
                    
                    # ตรวจสอบ Quick Replies
                    if 'quickReplies' in message:
                        quick_reply_items = []
                        if 'quickReplies' in message['quickReplies'] and isinstance(message['quickReplies']['quickReplies'], list):
                            for qr in message['quickReplies']['quickReplies']:
                                quick_reply_items.append(
                                    QuickReplyItem(
                                        action=MessageAction(
                                            label=qr[:20],  # LINE จำกัดความยาวของป้ายชื่อไม่เกิน 20 ตัวอักษร
                                            text=qr
                                        )
                                    )
                                )
                        if quick_reply_items:
                            quick_replies = QuickReply(items=quick_reply_items)
                        
                    # ตรวจสอบ Custom Payload (เช่น Flex Message)
                    if 'payload' in message:
                        has_payload = True
                        process_payload(message['payload'], messages_to_reply)
            
            # ถ้าไม่มีข้อความใน fulfillmentMessages ให้ใช้ fulfillmentText (ถ้ามี)
            if not messages_to_reply and not has_payload and 'fulfillmentText' in response_dict['queryResult']:
                text_response = response_dict['queryResult']['fulfillmentText']
                if text_response and text_response not in INVALID_DIALOGFLOW_RESPONSES:
                    text_message = TextMessage(text=text_response)
                    messages_to_reply.append(text_message)
            
            # หากมีอย่างน้อยหนึ่งข้อความให้ส่งทั้งหมด
            if messages_to_reply:
                logger.info("พบคำตอบจาก Dialogflow ส่งคำตอบให้ผู้ใช้")
                # เพิ่ม quick replies ให้กับข้อความสุดท้าย (ถ้ามี)
                if quick_replies and messages_to_reply:
                    messages_to_reply[-1].quick_reply = quick_replies
                
                send_multiple_messages(line_bot_api, event.reply_token, messages_to_reply)
            else:
                # ขั้นตอนที่ 2: ค้นหาในเอกสาร
                logger.info("เริ่มขั้นตอนที่ 2: ค้นหาในเอกสาร")
                reply_text, found_in_docs, rag_context = search_from_documents(actual_message)

                # ส่งข้อความที่ได้
                text_message = TextMessage(text=reply_text)
                if quick_replies:
                    text_message.quick_reply = quick_replies
                
                send_multiple_messages(line_bot_api, event.reply_token, [text_message])
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อความ: {str(e)}")
        send_text_message(line_bot_api, event.reply_token, "ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)