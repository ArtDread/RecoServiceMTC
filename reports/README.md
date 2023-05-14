# Report

## Bug tracking

![All errors presented](./Bug_tracking/sentry_all_errors.png)

Все ошибки сервиса отображались автоматически с отслеживанием кроме одной: `HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Not authenticated")`, которая отмечена в баг-трекере как `<unlabeled event>`. Данный *Exception* хэндлится FastAPI автоматически в классe `HTTPBearer`, но **sentry-sdk** фиксирует ее неверно. По сути баг
(в интернете есть похожие случаи, но при интеграции **Sentry** с другими сервисами)

Примечание:\
NameError было вызвано вручную при ошибке в коде, `LocalProtocolError` и `can't send data when our state is ERROR` неизбежно всплывали при опрашивании сервиса ботом с профайлером спустя время.

![Multiple Error Emulation](./Bug_tracking/sentry_emulation_error.png)

Эмуляция случайной ошибки, которая всплывает для пользователей с user_id кратным 666. Сообщение ошибки, описание и путь HTTP запроса зафиксированы трекером.

## Profiles

![Profile, health](./Profiles/profile_health.svg)

Ожидаемо, при старте сервера большую часть времени занимает загрузка данных и инициализация всех моделей:

- popular in category model ~ **38.18 %** (заметно долго, очень тяжелая модель)
- online knn & offline knn model ~ 8.45 %, 11.94 % ~ **20.39 %**
- lightfm model 1 & lightfm model 2 & ann light fm ~ 8.19 %, 4.80 %, 13.96 %, ~ **26.95 %**

Итого `85.52 %` времени занимает инстанциирование рабочих моделей при запуске сервера.

Глубоких стаков немного (в основном это импорты различных классов), и все коллы в них быстрые, особенно относительно загрузки данных и инициализации моделей.

![Profile, light reco model](./Profiles/profile_light.svg)

![Profile, heavy reco model](./Profiles/profile_hard.svg)

При опрашивании сервиса ботом для любой модели и запущенным профайлером, он практически мгновенно падал с вышеупомянутой ошибкой (сервер видимо не тянул). Удалось обработать мизерное количество пользователей, поэтому получилась скорее вырезка данных.

Тем не менее, можно проанализирвоать стак для тяжелой модели (online knn), связанный с получением RecoResponse для 990 пользователей. Видно, что топ колл идет от middleware sentry sdk, чтобы отлавливались ошибки в нижеследующих коллах.

![Profile, heavy reco model, zoom on recos](./Profiles/profile_hard_zoom_recos.png)

Непосредственно стак, имеющий отношение к работе sentry-sdk, на сервере за время работы тяжелой модели

![Profile, heavy reco model, zoom on sentry](./Profiles/profile_hard_zoom_sentry.png)

Для легкой модели картина немного другая, ошибки там посыпались раньше (видимо из-за скорости запросов), поэтому стак с коллами генерации рекомендаций выглядит блекло

![Profile, light reco model, zoom on recos](./Profiles/profile_light_zoom_recos.png)

А вот стак sentry-sdk в этот раз переполнен большим количеством коллов

![Profile, light reco model, zoom on sentry](./Profiles/profile_light_zoom_sentry.png)

Очевидно, что если бы не эта ошибка, то профили для легкой и тяжелой моделей рекомендаций отличались бы разительно меньше, теоретически только длина столбиков топ коллов, связанных с запросом рекомендаций, отличалась бы из-за разницы во времени для разных моделей.
