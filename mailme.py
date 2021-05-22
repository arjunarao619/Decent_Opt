#!/usr/bin/python

import smtplib

import argparse



def sendmes(mes):
    sender = 'arjunaao619@gmail.com'
    receivers = ['arjunrao@link.cuhk.edu.hk']

    message = """From: From Person <from@fromdomain.com>
    To: To Person <to@todomain.com>
    Subject: Batch Job Completed Arjun

    
    """ + mes

    try:
        smtpObj = smtplib.SMTP('localhost')
        smtpObj.sendmail(sender, receivers, message)         
        print("Successfully sent email")
    except SMTPException:
        print("Error: unable to send email")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send periodic email updates to Arjun")
    parser.add_argument("--message", default="Job Completed", type=str)
    conf = parser.parse_args()
    sendmes(conf.message)