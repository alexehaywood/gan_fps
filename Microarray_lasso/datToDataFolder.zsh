dirs=($(ls | sed -e '/^[0-9].*$/!d'));
for exp in $dirs; do
	data=($(ls $exp | sed -e '/^dat\_.*$/!d'));
	for x in $data; do
		mv "${exp}/${x}" "${exp}/Data/${x}";
	done;
done;
