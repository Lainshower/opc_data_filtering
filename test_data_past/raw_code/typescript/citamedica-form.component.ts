import { Component, OnInit, Inject, ViewEncapsulation } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { MAT_DIALOG_DATA, MatDialogRef, MatDialog } from '@angular/material';
import { debounceTime, tap, switchMap, finalize } from 'rxjs/operators';
import { format } from 'date-fns';  
import { EntidadService } from 'app/services/entidad.service';
import { EstadoService } from 'app/services/estado.service';
import { ReferenciaService } from 'app/services/referencia.service';
import { SeguroplanService } from 'app/services/seguroplan.service';
import { ProductoService } from 'app/services/producto.service';
import { CitamedicaService } from 'app/services/citamedica.service';
import { SedeService } from 'app/services/sede.service';
import { Utils } from 'app/services/utils'; 
import { CitamedicaModel } from './citamedica.model';
import { EntidadFormComponent } from 'app/components/entidad/entidad-form/entidad-form.component';

@Component({
  selector: 'app-citamedica-form',
  templateUrl: './citamedica-form.component.html',
  styleUrls: ['./citamedica-form.component.scss'],
  encapsulation: ViewEncapsulation.None
})
export class CitamedicaFormComponent implements OnInit {

  dialogRef: any;
    action: string;
    citamedica: any;
    citamedicaForm: FormGroup;
    citamedicaResponse: any;
    dialogTitle: string;
    estadoscita: any[] = [{ idestadodocumento: 4, nombre: 'Pendiente' }, { idestadodocumento: 5, nombre: 'Confirmada' }];
    referencias: any[] = [];
    atenciones: any[] = [];
    segurosplanes: any[] = [];
    productos: any[] = [];
    sedes: any[] = [];

    disponibles: any[] = []; // Médicos y horas
    disponibleMedicos: any[] = []; // Solo médicos
    disponibleHoras: any[] = []; // Solo horas

    searchResult: any[] = [];
    isLoadingSearch = false;

    idsededefault: number;
    timedefault: number;
    clientedefault: any;
    newPaciente = true;

    submitted = false;

    constructor(
        public matDialogRef: MatDialogRef<CitamedicaFormComponent>,
        @Inject(MAT_DIALOG_DATA) private _data: any,
        private _entidadService: EntidadService,
        private _estadoService: EstadoService,
        private _referenciaService: ReferenciaService,
        private _seguroplanService: SeguroplanService,
        private _productoService: ProductoService,
        private _citamedicaService: CitamedicaService,
        private _sedeService: SedeService,
        private _utils: Utils,
        private _matDialog: MatDialog,
    ) {
        this.action = _data.action;
        this.idsededefault = _data.idsede || '';
        this.timedefault = _data.time || '';
        this.clientedefault = _data.cliente || '';

        console.log(this.idsededefault, this.timedefault, this.clientedefault);

        if (this.action === 'edit') {
            this.dialogTitle = 'Modificar cita';
            this.show();
        } else {
            this.dialogTitle = 'Nueva cita';
        }
    }

    ngOnInit(): void {

        this.citamedicaForm = this.createEventForm();
        this.indexReferencias();
        this.indexAtenciones();
        this.indexSegurosplanes();
        this.indexProductos();
        this.indexSedes();

        if (this.action === 'new') {
            this.citamedicaForm.controls['idestado'].setValue(4); // Pendiente
        }

        this.citamedicaForm.controls['cliente'].valueChanges
            .pipe(
                debounceTime(300),
                tap(() => this.isLoadingSearch = true),
                switchMap(data => {
                    if (typeof data === 'object') {
                        this.newPaciente = false;
                    } else {
                        this.newPaciente = true;
                    }

                    if (data && typeof data === 'string') {
                        const param = {
                            idempresa: this._entidadService.usuario.idempresa,
                            page: 1,
                            pageSize: 10,
                            orderName: 'entidad',
                            orderSort: 'asc',
                            likeEntidad: data
                        };
                        return this._entidadService.index(param)
                            .pipe(
                                finalize(() => this.isLoadingSearch = false),
                            );
                    } else {
                        this.isLoadingSearch = false;
                        return [];
                    }
                })
            )
            .subscribe((data: any) => {
                this.searchResult = data;
            });

        // Tanto fecha como idsede  invocan cita disponibles
        this.citamedicaForm.controls['fecha'].valueChanges
            .subscribe(data => {
                console.log('fecha valueChanges', data);
                if (data) {
                    this.citasDisponibles();
                }
            });

        // this.citamedicaForm.controls['idsede'].valueChanges
        //     .subscribe(data => {
        //         console.log('idsede valueChanges', data);
        //         if (data) {
        //             this.citasDisponibles();
        //         }
        //     });

        this.citamedicaForm.controls['inicio'].valueChanges
            .subscribe(data => {
                this.citamedicaForm.controls['idmedico'].setValue('');
                this.disponibleMedicos = [];
                if (data) {
                    console.log('inicio valueChanges', data, this.disponibles);
                    this.disponibles.forEach(item => {
                        item.horas.forEach(hora => {
                            if (hora.inicio === data) {
                                this.disponibleMedicos.push({
                                    idmedico: item.idmedico,
                                    entidad: item.nombre,
                                });
                            }
                        });
                    });

                    if (this.disponibleMedicos.length === 1) {
                        this.citamedicaForm.controls['idmedico'].setValue(this.disponibleMedicos[0].idmedico);
                    }
                }
            });
    }

    displayFn(user?): string | undefined {
        return user ? user.entidad : undefined;
    }

    createEventForm(): FormGroup {
        // sede = new FormControl({ value: 'OSI CHACARILLA', disabled: true }, [Validators.required]);
        return new FormGroup({
            idcitamedica: new FormControl(''),
            idsede: new FormControl('', [Validators.required]),
            fecha: new FormControl(this.timedefault || '', [Validators.required]),
            inicio: new FormControl('', [Validators.required]),
            idmedico: new FormControl('', [Validators.required]),
            idestado: new FormControl('', [Validators.required]),
            idreferencia: new FormControl('', [Validators.required]),
            idatencion: new FormControl('', [Validators.required]),
            idaseguradoraplan: new FormControl(''),
            idproducto: new FormControl(''),
            descripcion: new FormControl(''),
            costocero: new FormControl(false),
            validation_fechavencida: new FormControl(false),
            cliente: new FormControl({ value: this.clientedefault || '', disabled: this.action === 'edit' }, [Validators.required])
        });
    }

    citasDisponibles(): void {
        setTimeout(() => {
            let param;
            param = Object.assign({}, this.citamedicaForm.value);
            param.fecha = format(param.fecha, 'YYYY-MM-DD');

            this.disponibles = [];
            this.disponibleHoras = [];
            this._citamedicaService.disponibilidad(param.idsede, { fecha: param.fecha })
                .subscribe((data: any): void => {
                    this.disponibles = this.disponibilidad(data.data);
                    this.disponibleHoras = this.disponibilidadHoras(this.disponibles);

                    // Valores por defecto
                    if (this.action === 'new' && this.disponibleMedicos.length === 1) {
                        this.citamedicaForm.controls['idmedico'].setValue(this.disponibleMedicos[0].idmedico);
                    }

                    if (this.action === 'new' && this.timedefault) {
                        const inicio = format(this.timedefault, 'HH:mm:ss');
                        this.disponibleHoras.forEach(item => {
                            if (inicio === item.idhora) {
                                this.citamedicaForm.controls['inicio'].setValue(inicio);
                            }
                        });
                    }

                    if (this.action === 'new' && this.disponibleHoras.length === 1 && this.citamedicaForm.controls['inicio'].value === '') {
                        this.citamedicaForm.controls['inicio'].setValue(this.disponibleHoras[0].idhora);
                    }
                });
        }, 100);
    }

    // Mostrar cita médica
    show(): void {
        const param = {
            conRecurso: 'paciente, medico'
        };
        this._citamedicaService.show(this._data.citamedica.idcitamedica, param)
            .subscribe((data: any) => {

                this.citamedicaResponse = Object.assign({}, data.data);

                let citamedica: any;
                citamedica = Object.assign({}, data.data);
                citamedica.fecha = this._utils.convertDate(citamedica.fecha);
                citamedica.cliente = { identidad: citamedica.paciente.identidad, entidad: citamedica.paciente.entidad };
                citamedica.validation_fechavencida = false;
                citamedica.costocero = citamedica.costocero === '1' ? true : false;

                this.citamedicaForm.setValue(new CitamedicaModel(citamedica));

                // Añadir hora a listado 
                if (!this._utils.existValue(this.disponibleHoras, 'idhora', this.citamedicaResponse.inicio)) {
                    console.log('Existe hora show');
                    this.disponibleHoras.push({
                        idhora: this.citamedicaResponse.inicio,
                        nombre: this._utils.convertDate(this.citamedicaResponse.fecha, this.citamedicaResponse.inicio)
                    });
                }

                // Añadir médico a listado 
                if (!this._utils.existValue(this.disponibleMedicos, 'idmedico', this.citamedicaResponse.idmedico)) {
                    console.log('Existe medico show');
                    this.disponibleMedicos.push({
                        idmedico: this.citamedicaResponse.idmedico,
                        entidad: this.citamedicaResponse.medico.entidad
                    });
                }

            });
    }

    // Guardar cambios de formulario
    save(): void {
        let param;
        param = this.citamedicaForm.getRawValue();

        const fecha = format(param.fecha, 'YYYY-MM-DD');
        const fin = format(this._utils.sumDate(this._utils.convertDate(fecha, param.inicio), '00:14:00'), 'HH:mm:ss');

        param.fecha = fecha;
        param.fin = fin;
        param.costocero = param.costocero ? '1' : '0';
        param.validation = { fechavencida: param.validation_fechavencida ? '1' : '0', ncitaspendientes: '0' };

        this.submitted = true;

        if (this.action === 'new') {
            param.idpaciente = param.cliente.identidad;
            param.idestadopago = 72; // Pago pendiente  
            this._citamedicaService.create(param).subscribe((data) => {
                this.matDialogRef.close(data.data);
            }, error => {
                const message = this._utils.convertError(error.error.error);
                this.submitted = false;
                swal('Upss!', message, 'error');
            });
        }

        if (this.action === 'edit') {
            this._citamedicaService.update(param.idcitamedica, param).subscribe((data) => {
                this.matDialogRef.close(data.data);
            }, error => {
                const message = this._utils.convertError(error.error.error);
                this.submitted = false;
                swal('Upss!', message, 'error');
            });
        }
    }

    newEntidad(): void {
        this.dialogRef = this._matDialog.open(EntidadFormComponent, {
            panelClass: 'entidad-form-dialog',
            data: {
                action: 'new',
                dialogTitle: 'Nuevo paciente',
                tipo: 'cliente',
                tabPaciente: false
            },
            autoFocus: false
        });

        this.dialogRef.afterClosed()
            .subscribe((response: any) => {
                if (!response) {
                    return;
                }
                this.citamedicaForm.controls['cliente'].setValue({ identidad: response.identidad, entidad: response.entidad });
            });
    }

    editEntidad(entidad): void {
        this.dialogRef = this._matDialog.open(EntidadFormComponent, {
            panelClass: 'entidad-form-dialog',
            data: {
                action: 'edit',
                dialogTitle: 'Modificar paciente',
                tipo: 'cliente',
                tabPaciente: false,
                entidad: Object.assign({}, entidad)
            },
            autoFocus: false
        });

        this.dialogRef.afterClosed()
            .subscribe(response => {
                if (!response) {
                    return;
                }
                this.citamedicaForm.controls['cliente'].setValue({ identidad: response.identidad, entidad: response.entidad });
            });
    }

    // Listado de referencias para cita
    indexReferencias(): void {

        const param = {
            idempresa: this._entidadService.usuario.idempresa,
            orderName: 'nombre',
            orderSort: 'asc'
        };

        this._referenciaService.index(param)
            .subscribe((data: any) => {
                this.referencias = data.data;
            });
    }

    // Listado de atenciones para cita
    indexAtenciones(): void {

        const param = {
            orderName: 'nombre',
            orderSort: 'asc',
            tipo: 5
        };

        this._estadoService.index(param)
            .subscribe((data: any) => {
                this.atenciones = data.data;
            });
    }

    // Listado de planes de seguros
    indexSegurosplanes(): void {
        const param = {
            orderName: 'nombre',
            orderSort: 'asc',
            conRecurso: 'seguro',
            cubierto: '1',
            reservacita: '1'
        };

        this._seguroplanService.index(param)
            .subscribe((data: any) => {
                this.segurosplanes = data.data;
            });
    }

    // Listado de productos
    indexProductos(): void {
        const param = {
            idempresa: this._entidadService.usuario.idempresa,
            orderName: 'nombre',
            orderSort: 'asc',
            idtipoproducto: 2,
            tratamientoind: '1'
        };

        this._productoService.index(param)
            .subscribe((data: any) => {
                this.productos = data.data;
            });
    }

    // Listado de sedes
    indexSedes(): void {
        const param = {
            idempresa: this._entidadService.usuario.idempresa,
            orderName: 'nombre',
            orderSort: 'asc',
            comercial: '1'
        };

        this._sedeService.index(param)
            .subscribe((data: any) => {
                this.sedes = data.data;
                if (this.action === 'new' && this.idsededefault) {
                    this.citamedicaForm.controls['idsede'].setValue(this.idsededefault);
                }

                if (this.action === 'new' && this.timedefault) {
                    this.citasDisponibles();
                }
            });
    }

    // Añadir hora de cita en caso de no existir en diponibilidad
    private disponibilidad(data): any[] {

        if (this.action === 'edit' && this.citamedicaResponse) {
            let existemedico = false;
            let existehora = false;
            let indice = 0;

            data.forEach((item, index) => {

                if (item.idmedico === this.citamedicaResponse.idmedico) {
                    existemedico = true;
                    indice = index;
                }

                item.horas.forEach(hora => {

                    if (hora.idmedico === this.citamedicaResponse.idmedico &&
                        hora.fecha === this.citamedicaResponse.fecha &&
                        hora.inicio === this.citamedicaResponse.inicio
                    ) {
                        existehora = true;
                    }
                });
            });

            const horacita = {
                idmedico: this.citamedicaResponse.idmedico,
                nombre: this.citamedicaResponse.medico.entidad,
                fecha: this.citamedicaResponse.feca,
                inicio: this.citamedicaResponse.inicio,
                fin: this.citamedicaResponse.fin
            };

            if (existemedico && !existehora) {
                data[indice].horas.push(horacita);
            }

            if (!existemedico && !existehora) {
                data.push({
                    idmedico: this.citamedicaResponse.idmedico,
                    nombre: this.citamedicaResponse.medico.entidad,
                    horas: [horacita],
                });
            }
        }

        return data;
    }

    private disponibilidadHoras(data): any[] {

        const horas = [];
        const fecha = format(this.citamedicaForm.controls['fecha'].value, 'YYYY-MM-DD');

        // Todas las horas
        let tmphorasdisp: any[] = [];

        const tmpHoras: string[] = data.map(row => row.horas.map(item => item.inicio));
        tmpHoras.forEach(val => {
            tmphorasdisp = tmphorasdisp.concat(val);
        });

        // Unique y Date
        tmphorasdisp = tmphorasdisp
            .filter((x, i, a) => x && a.indexOf(x) === i)
            .map(val => {
                return this._utils.convertDate(fecha, val);
            });

        // Ordenamiento asc
        tmphorasdisp.sort((a, b) => {
            return a - b;
        });

        // Array de objetos
        for (let index = 0; index < tmphorasdisp.length; index++) {
            const val = tmphorasdisp[index];
            horas.push({
                idhora: format(val, 'HH:mm:ss'),
                nombre: val
            });
        }

        return horas;
    }

}
