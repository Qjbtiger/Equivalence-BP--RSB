import logging
import requests
import time

class QLog(object):
    def __init__(self):
        pass

    def set(self, stream=True, file=False, fileLogName='1', fileMode='a'):
        self.logger = logging.getLogger()
        self.logger.setLevel('INFO')
        basicFormat = '[%(asctime)s][%(levelname)s] %(message)s'
        dateFormat = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(basicFormat, dateFormat)
        if stream:
            chlr = logging.StreamHandler() # 输出到控制台的handler
            chlr.setFormatter(formatter)
            self.logger.addHandler(chlr)
        if file:
            fhlr = logging.FileHandler('./log/{}.log'.format(fileLogName), mode=fileMode) # 输出到文件的handler, add mode
            fhlr.setFormatter(formatter)
            self.logger.addHandler(fhlr)

        self.isRegisterPushDeer = False
        self.isEnablePushDeer = False

    def info(self, text=''):
        self.logger.info(text)

    def weakInfo(self, text=''):
        print('[{}][WEAKINFO] {}\r'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), text), end='')

    def registerPushDeer(self, key):
        self.isRegisterPushDeer = True
        self.pushDeerKey = key
        self.isEnablePushDeer=True

    def disablePushDeer(self):
        self.isEnablePushDeer = False

    def pushDeer(self, text):
        self.info(text)
        if self.isRegisterPushDeer and self.isEnablePushDeer:
            text = '[{}][INFO] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), text)
            requests.post('https://api2.pushdeer.com/message/push?pushkey={}&text={}'.format(self.pushDeerKey, text))

qLog = QLog()