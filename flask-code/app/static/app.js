const chkbxAll=document.getElementsByClassName('select_all')
const chkbxOptions=document.getElementsByClassName('select-option')

function selectAllChkboxes() {
	const isChecked = chkbxAll.chkbxAll.checked;
	
	for (let i=0;i<chkbxOptions.length; i++) {
			chkbxOptions[i].checked = isChecked;
		}
}

console.log(chkbxOptions)