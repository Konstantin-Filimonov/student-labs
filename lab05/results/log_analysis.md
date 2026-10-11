# Индивидуальное задание. Анализ журнала `sample_auth.txt`

Строк в журнале: 27, находок: 4, время: 20.29 с, JSON-схема сервера: да, ошибок разбора: 0

| Severity | Категория | Доказательство | Причина | Рекомендация |
|---|---|---|---|---|
| high | Unauthorized Access Attempts | Failed password for root from 203.0.113.45 port 51122 ssh2, Failed password for root from 203.0.113.45 port 51126 ssh2, etc. | Multiple failed login attempts to the root account from an unknown IP address (203.0.113.45) over SSH, which is a common indicator of brute force or unauthorized access attempts. | Implement multi-factor authentication for root and review logs for any suspicious activity. |
| high | Unauthorized Access Attempts | Failed password for ivanov from 10.0.2.34 port 53390 ssh2, Accepted password for ivanov from 10.0.2.34 port 53392 ssh2 | Multiple failed login attempts to the user 'ivanov' and then a successful login attempt from an unknown IP address (10.0.2.34) over SSH, which is suspicious. | Implement multi-factor authentication for ivanov and review logs for any suspicious activity. |
| high | Unauthorized Access Attempts | [UFW BLOCK] IN=eth0 SRC=198.51.100.7 DST=10.0.0.5 PROTO=TCP SPT=44000 DPT=21, [UFW BLOCK] IN=eth0 SRC=198.51.100.7 DST=10.0.0.5 PROTO=TCP SPT=44001 DPT=22, etc. | Multiple blocked incoming connections from an unknown IP address (198.51.100.7) to the server's SSH and FTP ports, which is a common indicator of unauthorized access attempts. | Implement multi-factor authentication for all users and review logs for any suspicious activity. |
| high | Unauthorized Access Attempts | sudo: www-data : TTY=unknown ; PWD=/var/www/html ; USER=root ; COMMAND=/bin/cat /etc/shadow | A non-privileged user (www-data) attempting to access the root directory and viewing the shadow file, which is a sensitive system file. | Review sudo logs for any suspicious activity and ensure that www-data does not have unnecessary permissions. |

**Вывод модели:** Multiple unauthorized login attempts detected. Additionally, an attempt was made by a non-privileged user to access a sensitive system file (shadow).

**Проверка по эталону:** найдено 3 из 4 (recall 0.75); ложных срабатываний: 1

Пропущено: Создан пользователь support_tmp и добавлен в группу sudo сразу после входа

Ложные: Одна ошибка пароля штатного пользователя и штатные действия ivanov
