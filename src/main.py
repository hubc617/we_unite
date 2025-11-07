import sys
import os
import json
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, 
                             QWidget, QVBoxLayout, QFileDialog)  # 补充QFileDialog导入
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QSizePolicy
from ui_forms.ui_chat_interface import Ui_chat_interface as ChatUi
from ui_forms.ui_image_generator import Ui_MainWindow as ImageUi
from ui_forms.ui_history_manager import Ui_MainWindow as HistoryUi

import threading
from dotenv import load_dotenv  # 加载.env文件
from openai import OpenAI
from PyQt5.QtWidgets import QMessageBox 
from PyQt5.QtWidgets import QTextEdit
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PyQt5.QtCore import QThread, pyqtSignal



load_dotenv()



# 初始化OpenAI客户端
openai_client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ChatInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = ChatUi()
        self.ui.setupUi(self)
        self.ui.centralwidget.setLayout(self.ui.verticalLayout)
        self.ui.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.ui.verticalLayout.setSpacing(10)
        # 禁止聊天记录框编辑
        self.ui.chat_history.setReadOnly(True)

        # 添加点击清空输入框功能
        self.ui.text_input.mousePressEvent = self.on_input_click

        self.setup_connections()
    
    def on_input_click(self, event):
        """点击输入框时清空内容"""
        if self.ui.text_input.toPlainText().strip() == "输入消息...":
            self.ui.text_input.clear()
        # 确保正常的点击事件继续执行
        super(type(self.ui.text_input), self.ui.text_input).mousePressEvent(event)

    def setup_connections(self):
        # 绑定按钮事件
        self.ui.send_btn.clicked.connect(self.send_message)
        self.ui.clear_btn.clicked.connect(self.clear_chat)
        self.ui.text_input.keyPressEvent = self.on_text_input_key_press

    def on_text_input_key_press(self, event):
        """Enter发送消息，Ctrl+Enter换行，保留其他按键默认行为"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() == Qt.ControlModifier:
                # Ctrl+Enter：插入换行
                self.ui.text_input.insertPlainText("\n")
            else:
                # Enter：发送消息
                self.send_message()
            event.accept()
        else:
            # 其他按键（如Backspace）使用默认行为
            QTextEdit.keyPressEvent(self.ui.text_input, event)

        
    def clear_chat(self):
        self.ui.chat_history.clear()

    def send_message(self):
        user_msg = self.ui.text_input.toPlainText().strip()
        if not user_msg:
            QMessageBox.warning(self, "输入提示", "请输入消息内容后再发送！")
            return
        # 1. 显示用户消息
        self.ui.chat_history.append(f"<b>[{self.get_current_time()}]</b> 你: {user_msg}")
        self.ui.text_input.clear()
        # 禁用发送按钮，防止重复发送
        self.ui.send_btn.setEnabled(False)
        # 滚动到底部
        self.ui.chat_history.verticalScrollBar().setValue(
            self.ui.chat_history.verticalScrollBar().maximum()
        )

        # 2. 多线程调用OpenAI API（避免UI卡顿）
        threading.Thread(
            target=self.call_openai_api,
            args=(user_msg,),
            daemon=True  # 线程随主程序退出而关闭
        ).start()

    def call_openai_api(self, user_msg):
        """调用OpenAI API获取回复（子线程中执行）"""
        # 显示“思考中”提示（需用QMetaObject.invokeMethod切换到主线程更新UI）
        self.update_chat_history("<i>[AI正在思考...]</i>")

        try:
            # 调用OpenAI ChatCompletion API（gpt-3.5-turbo，性价比高）
            response = openai_client.chat.completions.create(
                model="deepseek-v3.2-exp",
                messages=[{"role": "user", "content": user_msg}],
                # 通过 extra_body 设置 enable_thinking 开启思考模式，该参数仅对 deepseek-v3.2-exp 和 deepseek-v3.1 有效。deepseek-v3 和 deepseek-r1 设定不会报错
                extra_body={"enable_thinking": True},
                stream=True,
                stream_options={
                    "include_usage": True
                },
            )
            #ai_msg = response.choices[0].message.content.strip()

            ai_msg = ""
            for chunk in response:  # 遍历每个流式片段
                if chunk.choices and chunk.choices[0].delta.content:
                    ai_msg += chunk.choices[0].delta.content  # 累加内容

            # 显示完整回复
            self.update_chat_history(f"<b>[{self.get_current_time()}]</b> AI: {ai_msg.strip()}")
            self.save_chat_to_history(user_msg, ai_msg.strip())


        except Exception as e:
            # 异常处理（网络错误、密钥错误等）
            error_msg = f"AI回复失败：{str(e)}"
            self.update_chat_history(f"<span style='color:red;'>{error_msg}</span>")
            QMessageBox.critical(self, "API错误", error_msg)

        finally:
            # 恢复发送按钮可用
            self.ui.send_btn.setEnabled(True)

    def update_chat_history(self, content):
        """更新聊天记录（需在主线程执行，避免UI线程安全问题）"""
        from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self.ui.chat_history,
            "append",
            Qt.QueuedConnection,
            Q_ARG(str, content)
        )
        # 滚动到底部（同样需主线程执行）
        QMetaObject.invokeMethod(
            self.ui.chat_history.verticalScrollBar(),
            "setValue",
            Qt.QueuedConnection,
            Q_ARG(int, self.ui.chat_history.verticalScrollBar().maximum())
        )

    def save_chat_to_history(self, user_msg, ai_msg):
        """保存聊天记录（含完整上下文，支持续聊）"""
        try:
            # 定义历史记录路径（使用绝对路径，避免相对路径问题）
            history_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../history.json"))
            print(f"保存历史到：{history_path}")  # 调试日志
            
            # 读取现有历史（文件不存在则初始化）
            history = []
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            
            # 构造完整对话上下文（用于续聊）
            chat_context = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ai_msg}
            ]
            
            # 添加新记录（包含摘要和上下文）
            history.append({
                "id": str(uuid.uuid4()),  # 唯一标识，用于续聊
                "time": self.get_current_time(),
                "type": "chat",
                "summary": f"你: {user_msg[:30]}...\nAI: {ai_msg[:30]}...",  # 摘要
                "context": chat_context  # 完整上下文
            })
            
            # 写入文件（确保目录存在）
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            print("聊天记录保存成功")  # 调试日志
            
        except Exception as e:
            error_msg = f"保存聊天历史失败：{str(e)}"
            print(error_msg)  # 打印错误日志
            QMessageBox.warning(self, "保存失败", error_msg)

    @staticmethod
    def get_current_time():
        """获取当前时间（格式：2024-05-20 15:30:45）"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        
    # def get_ai_response(self, prompt):
    #     # 模拟AI回复，实际项目中替换为真实调用
    #     import time
    #     time.sleep(1)  # 模拟网络延迟
    #     return f"这是对'{prompt}'的回复。在实际应用中，这里会显示AI模型生成的内容。"

class ImageGenerateThread(QThread):
    """图像生成子线程（避免UI卡顿）"""
    # 信号：生成进度（步数）、生成完成（图像路径）、生成失败（错误信息）
    progress_signal = pyqtSignal(int)
    finish_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, prompt, width, height, num_inference_steps):
        super().__init__()
        self.prompt = prompt
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.pipeline = None

    def run(self):
        """线程执行：加载模型+生成图像"""
        try:
            # 1. 加载Stable Diffusion模型（优先GPU，无GPU则用CPU）
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 使用DPMSolver调度器，生成速度比默认快
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                scheduler=scheduler,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)

            # 2. 生成图像（带进度反馈）
            def progress_callback(step, timestep, latents):
                """进度回调函数，发送当前步数"""
                self.progress_signal.emit(step + 1)  # step从0开始，+1后与总步数对应

            image = self.pipeline(
                prompt=self.prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                callback=progress_callback  # 绑定进度回调
            ).images[0]

            # 3. 保存图像到resources目录
            output_path = os.path.join("resources", f"sd_output_{self.get_current_time()}.png")
            image.save(output_path)
            self.finish_signal.emit(output_path)

        except Exception as e:
            self.error_signal.emit(str(e))

    @staticmethod
    def get_current_time():
        """生成唯一文件名（避免重复）"""
        return datetime.now().strftime("%Y%m%d%H%M%S")

class ImageGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = ImageUi()
        self.ui.setupUi(self)
        # 布局配置
        self.ui.centralwidget.setLayout(self.ui.horizontalLayout)  # horizontalLayout是Designer中中央部件的布局名
        self.ui.horizontalLayout.setContentsMargins(10, 10, 10, 10)
        self.ui.horizontalLayout.setSpacing(10)
        # 初始化变量
        self.current_pixmap = None
        self.generate_thread = None  # 图像生成线程
        # 设置参数范围
        self.ui.width_spin.setRange(256, 1024)    # 宽度：256-1024
        self.ui.height_spin.setRange(256, 1024)   # 高度：256-1024
        self.ui.steps_slider.setRange(10, 50)     # 步数：10-50
        self.ui.steps_slider.setValue(20)         # 默认步数：20
        self.ui.steps_label.setText(f"生成步数：{20}")  # 显示当前步数（需在Designer中添加QLabel命名为steps_label）
        self.setup_connections()

    def setup_connections(self):
        self.ui.generate_btn.clicked.connect(self.start_generate)
        self.ui.save_btn.clicked.connect(self.save_image)
        self.ui.steps_slider.valueChanged.connect(self.update_steps_label)

    def update_steps_label(self, value):
        """更新生成步数显示"""
        self.ui.steps_label.setText(f"生成步数：{value}")

    def start_generate(self):
        """开始生成图像（启动子线程）"""
        prompt = self.ui.prompt_input.text().strip()
        width = self.ui.width_spin.value()
        height = self.ui.height_spin.value()
        steps = self.ui.steps_slider.value()

        # 输入校验
        if not prompt:
            QMessageBox.warning(self, "输入提示", "请输入图像描述（Prompt）！")
            return
        if self.generate_thread and self.generate_thread.isRunning():
            QMessageBox.information(self, "生成提示", "当前已有生成任务在运行，请稍候！")
            return

        # 初始化状态
        self.ui.image_preview.clear()
        self.ui.image_preview.setText(f"正在加载模型...（{('GPU加速' if torch.cuda.is_available() else 'CPU')}）")
        self.ui.generate_btn.setEnabled(False)
        self.ui.save_btn.setEnabled(False)
        # 重置进度条（需在Designer中添加QProgressBar命名为generate_progress）
        self.ui.generate_progress.setRange(0, steps)
        self.ui.generate_progress.setValue(0)

        # 启动生成线程
        self.generate_thread = ImageGenerateThread(prompt, width, height, steps)
        self.generate_thread.progress_signal.connect(self.update_generate_progress)
        self.generate_thread.finish_signal.connect(self.on_generate_finish)
        self.generate_thread.error_signal.connect(self.on_generate_error)
        self.generate_thread.start()

    def update_generate_progress(self, step):
        """更新生成进度条"""
        self.ui.generate_progress.setValue(step)
        self.ui.image_preview.setText(f"正在生成：{step}/{self.ui.steps_slider.value()} 步")

    def on_generate_finish(self, image_path):
        """生成完成：显示图像"""
        self.current_pixmap = QPixmap(image_path)
        if not self.current_pixmap.isNull():
            # 自适应缩放图像
            scaled_pixmap = self.current_pixmap.scaled(
                self.ui.image_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.image_preview.setPixmap(scaled_pixmap)
            # 保存到历史记录
            self.save_image_to_history(image_path)
        else:
            self.ui.image_preview.setText("生成失败：无法加载图像文件")

        # 恢复控件状态
        self.ui.generate_btn.setEnabled(True)
        self.ui.save_btn.setEnabled(True)

    def on_generate_error(self, error_msg):
        """生成失败：显示错误信息"""
        error_info = f"生成失败：{error_msg}"
        self.ui.image_preview.setText(error_info)
        QMessageBox.critical(self, "生成错误", error_info)
        # 恢复控件状态
        self.ui.generate_btn.setEnabled(True)

    def save_image(self):
        """保存生成的图像"""
        if not self.current_pixmap or self.current_pixmap.isNull():
            QMessageBox.warning(self, "保存提示", "暂无生成的图像可保存！")
            return

        # 弹出保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像",
            f"sd_output_{self.get_current_time()}.png",
            "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*.*)"
        )
        if file_path:
            if self.current_pixmap.save(file_path):
                QMessageBox.information(self, "保存成功", f"图像已保存到：\n{file_path}")
            else:
                QMessageBox.error(self, "保存失败", "无法保存图像，请检查路径权限！")

    def save_image_to_history(self, image_path):
        """保存图像生成记录到history.json"""
        try:
            history = []
            if os.path.exists("history.json"):
                with open("history.json", "r", encoding="utf-8") as f:
                    history = json.load(f)
            # 添加记录（包含图像路径，方便后续查看）
            history.append({
                "time": self.get_current_time(),
                "type": "image",
                "content": f"描述：{self.ui.prompt_input.text().strip()}\n尺寸：{self.ui.width_spin.value()}x{self.ui.height_spin.value()}\n路径：{image_path}"
            })
            # 写入文件
            with open("history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存图像历史失败：{e}")

    @staticmethod
    def get_current_time():
        """获取当前时间（格式：2024-05-20 15:30:45）"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def resizeEvent(self, event):
        """窗口缩放时，更新图像预览"""
        super().resizeEvent(event)
        if self.current_pixmap and not self.current_pixmap.isNull():
            self.update_image_preview()

    def update_image_preview(self):
        """更新图像预览（自适应窗口大小）"""
        scaled_pixmap = self.current_pixmap.scaled(
            self.ui.image_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.ui.image_preview.setPixmap(scaled_pixmap)

class HistoryManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = HistoryUi()
        self.ui.setupUi(self)
        # 布局配置
        self.ui.centralwidget.setLayout(self.ui.horizontalLayout)  # horizontalLayout是Designer中中央部件的布局名
        self.ui.horizontalLayout.setContentsMargins(10, 10, 10, 10)
        self.ui.horizontalLayout.setSpacing(10)
        # 初始化变量
        self.history_data = []
        self.setup_connections()
        self.load_history()

    def setup_connections(self):
        self.ui.history_list.itemClicked.connect(self.show_history_detail)
        self.ui.delete_btn.clicked.connect(self.delete_history_item)
        self.ui.export_btn.clicked.connect(self.export_history)
        self.ui.search_input.textChanged.connect(self.search_history)  # 搜索功能（需在Designer中添加QLineEdit命名为search_input）

    def load_history(self):
        """加载历史记录"""
        self.ui.history_list.clear()
        self.history_data = []
        try:
            if os.path.exists("history.json"):
                with open("history.json", "r", encoding="utf-8") as f:
                    self.history_data = json.load(f)
            # 倒序显示（最新的记录在最上面）
            for item in reversed(self.history_data):
                item_type = "📝 对话" if item["type"] == "chat" else "🖼️ 图像"
                self.ui.history_list.addItem(f"{item_type} | {item['time']}")
        except Exception as e:
            self.ui.history_preview.setText(f"加载历史失败：{str(e)}")

    def show_history_detail(self, item):
        """显示选中记录的详情"""
        # 获取选中记录的索引（倒序显示，需反向计算）
        index = len(self.history_data) - 1 - self.ui.history_list.row(item)
        if 0 <= index < len(self.history_data):
            data = self.history_data[index]
            # 格式化显示详情
            detail = f"""
            <b>时间：</b>{data['time']}
            <br><b>类型：</b>{'对话记录' if data['type'] == 'chat' else '图像生成记录'}
            <br><b>内容：</b>
            <br><pre style='background-color:#f5f5f5; padding:8px; border-radius:4px;'>{data['content']}</pre>
            """
            self.ui.history_preview.setHtml(detail)
            # 若为图像记录，尝试显示缩略图（需在Designer中添加QLabel命名为image_thumbnail）
            if data["type"] == "image" and "路径：" in data["content"]:
                # 提取图像路径
                path_line = [line for line in data["content"].split("\n") if "路径：" in line][0]
                image_path = path_line.split("：")[-1].strip()
                if os.path.exists(image_path):
                    thumbnail = QPixmap(image_path).scaled(
                        self.ui.image_thumbnail.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.ui.image_thumbnail.setPixmap(thumbnail)
                else:
                    self.ui.image_thumbnail.setText("图像文件已删除")
            else:
                self.ui.image_thumbnail.clear()

    def delete_history_item(self):
        """删除选中的历史记录"""
        current_item = self.ui.history_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "删除提示", "请先选中要删除的历史记录！")
            return

        # 确认删除
        confirm = QMessageBox.question(
            self, "删除确认", "确定要删除这条历史记录吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        # 删除数据并更新文件
        index = len(self.history_data) - 1 - self.ui.history_list.row(current_item)
        self.history_data.pop(index)
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(self.history_data, f, ensure_ascii=False, indent=2)
        # 重新加载显示
        self.load_history()
        self.ui.history_preview.clear()
        self.ui.image_thumbnail.clear()
        QMessageBox.information(self, "删除成功", "选中的历史记录已删除！")

    def export_history(self):
        """导出所有历史记录"""
        if not self.history_data:
            QMessageBox.warning(self, "导出提示", "暂无历史记录可导出！")
            return

        # 弹出导出对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出历史记录",
            f"ai_history_{self.get_current_time()}.txt",
            "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*.*)"
        )
        if not file_path:
            return

        try:
            if file_path.endswith(".json"):
                # 导出为JSON格式（保留原始结构）
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.history_data, f, ensure_ascii=False, indent=2)
            else:
                # 导出为可读文本格式
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"AI助手历史记录（导出时间：{self.get_current_time()}）\n")
                    f.write("=" * 50 + "\n\n")
                    for i, item in enumerate(reversed(self.history_data), 1):
                        f.write(f"【{i}】{item['time']} | {'对话' if item['type'] == 'chat' else '图像'}\n")
                        f.write(f"内容：\n{item['content']}\n")
                        f.write("-" * 30 + "\n\n")
            QMessageBox.information(self, "导出成功", f"历史记录已导出到：\n{file_path}")
        except Exception as e:
            QMessageBox.error(self, "导出失败", f"导出历史记录出错：{str(e)}")

    def search_history(self, search_text):
        """搜索历史记录（按关键词匹配）"""
        if not search_text:
            # 搜索为空，重新加载所有记录
            self.load_history()
            return

        # 筛选包含关键词的记录
        filtered_data = [
            item for item in self.history_data
            if search_text.lower() in item["content"].lower() or
               search_text.lower() in item["time"].lower()
        ]
        # 更新列表显示
        self.ui.history_list.clear()
        for item in reversed(filtered_data):
            item_type = "📝 对话" if item["type"] == "chat" else "🖼️ 图像"
            self.ui.history_list.addItem(f"{item_type} | {item['time']}")

    @staticmethod
    def get_current_time():
        """获取当前时间（格式：20240520153045）"""
        return datetime.now().strftime("%Y%m%d%H%M%S")



class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI创作助手")
        self.setGeometry(100, 100, 1200, 800)  # 增大默认窗口尺寸，提升体验
        
        # 1. 配置QTabWidget为“自适应扩展”
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(
            QSizePolicy.Expanding,  # 水平方向：占满可用空间
            QSizePolicy.Expanding   # 垂直方向：占满可用空间
        )
        self.setCentralWidget(self.tabs)
        
        # 2. 初始化子窗口（对话、图像、历史）
        self.chat_interface = ChatInterface()
        self.image_generator = ImageGenerator()
        self.history_manager = HistoryManager()
        
        # 3. 关键：将子窗口的中央部件作为标签页内容（而非直接用QMainWindow）
        # 原因：QMainWindow嵌套时，直接添加会保留其边框/留白，用中央部件可消除
        self.tabs.addTab(self.chat_interface.centralWidget(), "对话助手")
        self.tabs.addTab(self.image_generator.centralWidget(), "图像生成")
        self.tabs.addTab(self.history_manager.centralWidget(), "历史记录")

if __name__ == "__main__":


    load_dotenv()


    app = QApplication(sys.argv)
    # 设置中文字体
    font = QFont("SimHei")
    app.setFont(font)
    window = MainApplication()
    window.show()
    sys.exit(app.exec_())
