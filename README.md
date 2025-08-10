# GenClime - модуль для прогноза климата

## Описание

Здесь весь backend модуля GenClimate, обернутый в микросервис на FastAPI и запускающийся в Docker.
Модуль предназначен для получения прогнозов на некоторое количество дней вперед. Пользователь может посмотреть прогноз по какому-либо городу или по каким-лиюо координатам (широте и долготе) на 15 дней по одной из моделей, а также выбрать определенный регион (в примерах - Япония), получить визуализацию прогноза на сетке по каждому из параметров или скачать прогноз в формате `.zarr`.

## Как это все работает

![Архитектура](imgs/backend_scheme.svg)

Есть модель предсказатель - `Diviner`. Это docker-контейнер (отдельный микросервис, не входит сюда), который запускается раз в сутки с помощью планировщика `cron`. Обновляет прогнозы и падает. 

Сейчас по API можно получить самые свежие прогнозы. Сейчас примеры прогнозов на 15 дней можно получить по ссылке.

### Point forecast

Это первый endpoint. Что это такое: пользователь вводит в форму либо город, по которому хочет получить прогноз, либо координаты. Потом он вводит `days` - число дней, за которые он хочет получить прогноз. А также параметр `params` - по каким параметрам пользователь хочет получить прогноз.

У `params` есть три опции: 

| Опция       | Параметры                                                                       |
| ----------- | ------------------------------------------------------------------------------- |
| **simple**  | `2t`, `10u`, `10v`, `msl`                                                       |
| **surface** | `10u`, `10v`, `2d`, `2t`, `msl`, `skt`, `sp`, `tcw`, `lsm`, `z`, `slor`, `sdor` |
| **full**    | *(все доступные переменные)*                                                    |

**Примечания**:

* `2t` — температура воздуха на 2 м
* `10u`, `10v` — компоненты ветра на 10 м
* `msl` — давление на уровне моря
*  и тд.

Пример запроса по API

```bash
curl "http://127.0.0.1:8003/v1/point_forecast?city=Moscow&days=3&params=surface&mode=base&model=medium"
```

или

> (с точки зрения API - равнозначные запросы)

```bash
curl "http://127.0.0.1:8003/v1/point_forecast?city=lat=55.62&lon=37.606&days=3&params=surface&mode=base&model=medium"
```
Ответ - в формате `.json`. Пример лежит в папке [`example`](./example), в файле [`out_point`](./example/out_point.json)

### Region forecast

Это второй endpoint, который делает чуть более сложные манипуляции. Здесь мы хотим получить предсказание не по одной точке, а по целому региону (`box`, который ограничен `lat1`, `lat2`, `lon1`, `lon2`. Также прогнозы получается на `days` дней. Тоже происходит фильтрация по `params`. Но выходом уже является - файл, который пользователь может скачать, и визуализация в `plotly` в формате `html` в `json`-ответе.

Пример запроса для Японии:

```bash 
curl -G "http://127.0.0.1:8003/v1/region_forecast" \
  --data-urlencode "lat1=46.5" \
  --data-urlencode "lon1=128.0" \
  --data-urlencode "lat2=24.0" \
  --data-urlencode "lon2=147.5" \
  --data-urlencode "days=3" \
  --data-urlencode "params=simple" \
  --data-urlencode "model=medium"
```

Аналогично, но записать ответ в `json`-файл сразу:

```bash
curl -fsS "http://127.0.0.1:8003/v1/region_forecast?lat1=46.5&lon1=128&lat2=24&lon2=147.5&days=3&params=surface&model=medium" \
  | sudo tee out.json >/dev/null
```
Ответ - в формате `.json`. Пример лежит в папке [`example`](./example), в файле [`out_region`](./example/out_region.json)


Как посмотреть в `Jupyter notebook` (быстро) как будет выглядеть визуализация:

```python
import json
from pathlib import Path
from IPython.display import IFrame

with open("/mnt/local/tester/genclimate_raw/example/out_region.json", encoding="utf-8") as f: #путь до файла json
    data = json.load(f)

html = "<!doctype html><meta charset='utf-8'><title>Region forecast</title>" + data.get("preview_html","")
Path("preview.html").write_text(html, encoding="utf-8")

IFrame(src="preview.html", width="100%", height=800)
```

Как выглядит визуализация прогноза:

![Демонстрация](imgs/visual_x2.gif)

## Что дальше?

* Добавление новой модели для среднесрочного прогноза (45 дней)
* интеграция с хранилищем S3 (обсуждается)


