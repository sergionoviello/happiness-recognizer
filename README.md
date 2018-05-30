### Start

```
docker build -t happiness-recognizer .
docker run -v $(pwd):/app -it --rm -p 5000:5000 happiness-recognizer
```
