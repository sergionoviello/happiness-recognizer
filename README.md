### Start

```
docker build -t happiness-recognizer .
docker run -v $(pwd):/app -it --rm -p 5000:5000 happiness-recognizer
```


### download file from eb
scp -i key_file.pem ec2-user@ec2_IP_Address:/var/app/current/db.sqlite ~/Downloads


scp -i ~/.ssh/awskeypair.pem ec2-user@34.252.84.55:/var/app/current/db.sqlite ~/Downloads

scp -r -i ~/.ssh/awskeypair.pem ec2-user@34.252.84.55:/var/app/current/static/uploads ~/Downloads/uploads