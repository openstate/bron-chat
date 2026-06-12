#!/bin/bash
set -euo pipefail

MYSQL_CONTAINER="bron-chat-mysql-1"
NETWORK="bron-chat_bron-chat"
BACKUP_DIR="/home/projects/bron-chat/backups/mysql"

MYSQL_ROOT_PASSWORD=$(awk -F= '/^MYSQL_ROOT_PASSWORD=/{sub(/^[^=]+=/, ""); gsub(/^["'\'']|["'\'']$/, ""); print; exit}' /home/projects/bron-chat/.env)

mkdir -p "$BACKUP_DIR"

CMD="mysqldump -h $MYSQL_CONTAINER -uroot --password=$MYSQL_ROOT_PASSWORD \
  --all-databases --ignore-table=mysql.event --skip-lock-tables \
  | gzip > /backups/latest-mysqldump-daily.sql.gz ; \
cp -p /backups/latest-mysqldump-daily.sql.gz /backups/\$(date +%A)-mysqldump-daily.sql.gz"

docker run --rm --network "$NETWORK" -v "$BACKUP_DIR":/backups:rw mysql:8.0 bash -c "$CMD"
