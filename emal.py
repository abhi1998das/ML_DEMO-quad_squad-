import smtplib
password='gaurav1234'
by='gaurav.llh03@gmail.com'
def send_mail(content,to):
    mail=smtplib.SMTP('smtp.gmail.com',587)
    mail.ehlo()
    mail.starttls()
    mail.login(by,password)
    mail.sendmail(by,to,content)
    mail.close()

