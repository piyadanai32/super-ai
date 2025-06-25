import json
import logging
from linebot.v3.messaging import (
    TextMessage, FlexMessage, FlexContainer, ReplyMessageRequest,
    QuickReply, QuickReplyItem, MessageAction
)

logger = logging.getLogger(__name__)

def prepend_bot_name_for_group(text, is_group):
    """Helper function to add bot name prefix in group context"""
    if is_group:
        return f"@DMC Chatbot {text}"
    return text

def process_payload(payload, messages_list, is_group=False):
    try:
        logger.info(f"กำลังประมวลผล payload: {json.dumps(payload, indent=2, ensure_ascii=False)[:500]}")
        if 'line' in payload and isinstance(payload['line'], dict):
            line_content = payload['line']
            if 'type' in line_content and line_content['type'] == 'flex':
                flex_message = create_flex_message(line_content, is_group)
                if flex_message:
                    messages_list.append(flex_message)
                return
        if isinstance(payload, dict) and 'type' in payload and payload['type'] == 'flex':
            flex_message = create_flex_message(payload, is_group)
            if flex_message:
                messages_list.append(flex_message)
            return
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการประมวลผล payload: {str(e)}")

def modify_action_for_group(action, is_group):
    """Modify action text for group context"""
    if is_group and isinstance(action, dict):
        if 'type' in action and action['type'] == 'message':
            if 'text' in action:
                action['text'] = prepend_bot_name_for_group(action['text'], is_group)
    return action

def create_flex_message(flex_content, is_group=False):
    try:
        logger.info(f"กำลังสร้าง Flex Message: {json.dumps(flex_content)[:200]}...")
        if 'contents' in flex_content:
            flex_contents = flex_content['contents']
            
            # Modify actions in flex contents for group context
            def process_component(component):
                if isinstance(component, dict):
                    if 'action' in component:
                        component['action'] = modify_action_for_group(component['action'], is_group)
                    for key, value in component.items():
                        if isinstance(value, (dict, list)):
                            process_component(value)
                elif isinstance(component, list):
                    for item in component:
                        process_component(item)
            
            if is_group:
                process_component(flex_contents)
            
            if isinstance(flex_contents, dict):
                json.dumps(flex_contents)
            flex_container = FlexContainer.from_dict(flex_content['contents'])
            return FlexMessage(
                alt_text=flex_content.get('altText', 'Flex Message'),
                contents=flex_container
            )
        return None
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการสร้าง Flex Message: {str(e)}")
        return None

def send_multiple_messages(line_bot_api, reply_token, messages):
    try:
        if not messages:
            logger.warning("ไม่มีข้อความที่จะส่ง")
            return
        logger.info(f"กำลังส่ง {len(messages)} ข้อความ")
        reply_request = ReplyMessageRequest(
            reply_token=reply_token,
            messages=messages
        )
        line_bot_api.reply_message_with_http_info(reply_request)
        logger.info("ส่งข้อความสำเร็จ")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการส่งข้อความหลายรายการ: {str(e)}")
        try:
            send_text_message(line_bot_api, reply_token, "ขออภัย เกิดข้อผิดพลาดในการส่งข้อความ")
        except:
            logger.error("ไม่สามารถส่งข้อความสำรองได้")

def send_text_message(line_bot_api, reply_token, text):
    try:
        text = text if text else "ขออภัย ไม่พบข้อมูล"
        if len(text) > 4997:
            text = text[:4997] + "..."
        logger.info(f"กำลังส่งข้อความตอบกลับ: {text[:100]}...")
        reply_request = ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=text)]
        )
        line_bot_api.reply_message_with_http_info(reply_request)
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการส่งข้อความตัวอักษร: {str(e)}")
